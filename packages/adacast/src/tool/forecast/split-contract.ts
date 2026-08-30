import { createReadStream } from "fs"
import path from "path"

export type ForecastSplitInput = {
  split_mode?: "ratio" | "timestamp"
  train_ratio?: number
  val_ratio?: number
  test_ratio?: number
  train_end_timestamp?: string
  validation_end_timestamp?: string
}

export type FrozenForecastSplit = {
  split_mode: "ratio" | "timestamp"
  train_ratio: number
  val_ratio: number
  test_ratio: number
  train_end_timestamp?: string
  validation_end_timestamp?: string
}

function splitMode(input: ForecastSplitInput): "ratio" | "timestamp" {
  const hasRatio = input.train_ratio !== undefined || input.val_ratio !== undefined || input.test_ratio !== undefined
  const hasTimestamp = input.train_end_timestamp !== undefined || input.validation_end_timestamp !== undefined
  const inferred = hasTimestamp && !hasRatio ? "timestamp" : "ratio"
  return input.split_mode ?? inferred
}

function resolveRatioSplit(input: ForecastSplitInput): FrozenForecastSplit {
  if (input.train_end_timestamp !== undefined || input.validation_end_timestamp !== undefined) {
    throw new Error("Ratio split mode cannot include timestamp boundaries")
  }
  if (input.train_ratio === undefined || input.val_ratio === undefined || input.test_ratio === undefined) {
    throw new Error("Ratio split mode requires train_ratio, val_ratio, and test_ratio")
  }
  return {
    split_mode: "ratio",
    train_ratio: input.train_ratio,
    val_ratio: input.val_ratio,
    test_ratio: input.test_ratio,
  }
}

function parseTimestampMillis(value: string): number | undefined {
  const match = value
    .trim()
    .replace("T", " ")
    .match(/^(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:\s+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?$/)
  if (!match) return undefined

  const [, year, month, day, hour = "0", minute = "0", second = "0"] = match
  const millis = Date.UTC(Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute), Number(second))
  return Number.isFinite(millis) ? millis : undefined
}

type BoundaryMatch = {
  label: string
  normalized: string
  millis?: number
  exactCount: number
  exactIndex: number
  equivalentCount: number
  equivalentIndex: number
}

function createBoundaryMatch(boundary: string, label: string): BoundaryMatch {
  const normalized = boundary.trim()
  if (!normalized) throw new Error(`${label} must not be empty`)
  return {
    label,
    normalized,
    millis: parseTimestampMillis(normalized),
    exactCount: 0,
    exactIndex: -1,
    equivalentCount: 0,
    equivalentIndex: -1,
  }
}

function recordBoundaryMatch(match: BoundaryMatch, timestamp: string, millis: number, index: number): void {
  if (timestamp === match.normalized) {
    match.exactCount += 1
    match.exactIndex = index
  }
  if (match.millis !== undefined && millis === match.millis) {
    match.equivalentCount += 1
    match.equivalentIndex = index
  }
}

function resolveBoundaryIndex(match: BoundaryMatch): number {
  const count = match.exactCount > 0 ? match.exactCount : match.equivalentCount
  const index = match.exactCount > 0 ? match.exactIndex : match.equivalentIndex
  if (count === 0) {
    throw new Error(`${match.label} "${match.normalized}" was not found exactly once in the dataset time column`)
  }
  if (count > 1) {
    throw new Error(`${match.label} "${match.normalized}" is ambiguous because it appears multiple times in the dataset`)
  }
  return index
}

async function scanTimestampColumn(
  datasetPath: string,
  timeColumn: string,
  trainBoundary: BoundaryMatch,
  validationBoundary: BoundaryMatch,
): Promise<{ dataRowCount: number; timestampValidationError?: string }> {
  const stream = createReadStream(datasetPath, { encoding: "utf-8" })
  let header: string[] = []
  let readingHeader = true
  let timeIndex = -1
  let dataRowCount = 0
  let previousTimestamp = Number.NEGATIVE_INFINITY
  let timestampValidationError: string | undefined
  let columnIndex = 0
  let field = ""
  let rowTimeValue = ""
  let fieldStarted = false
  let fieldHasNonWhitespace = false
  let rowHasNonWhitespace = false
  let rowHasFields = false
  let quoted = false
  let quotePending = false
  let skipLf = false

  const retainField = () => readingHeader || columnIndex === timeIndex
  const append = (character: string) => {
    if (retainField()) field += character
    fieldStarted = true
    if (!/\s/u.test(character)) fieldHasNonWhitespace = true
  }
  const finishField = () => {
    if (readingHeader) header.push(field)
    else if (columnIndex === timeIndex) rowTimeValue = field
    rowHasNonWhitespace ||= fieldHasNonWhitespace
    rowHasFields = true
    columnIndex += 1
    field = ""
    fieldStarted = false
    fieldHasNonWhitespace = false
  }
  const finishRow = () => {
    finishField()
    if (rowHasNonWhitespace) {
      if (readingHeader) {
        header = header.map((value, index) => (index === 0 ? value.replace(/^\uFEFF/, "") : value).trim())
        timeIndex = header.indexOf(timeColumn)
        if (timeIndex < 0) throw new Error(`Time column "${timeColumn}" was not found in the dataset CSV header`)
        readingHeader = false
      } else {
        const rowNumber = dataRowCount + 2
        if (timeIndex >= columnIndex) {
          throw new Error(`Dataset row ${rowNumber} is missing the time column "${timeColumn}"`)
        }
        const timestamp = rowTimeValue.trim()
        const millis = parseTimestampMillis(timestamp)
        if (millis === undefined) {
          timestampValidationError ??= `Dataset row ${rowNumber} contains an invalid timestamp in "${timeColumn}"`
          if (timestamp === trainBoundary.normalized) {
            trainBoundary.exactCount += 1
            trainBoundary.exactIndex = dataRowCount
          }
          if (timestamp === validationBoundary.normalized) {
            validationBoundary.exactCount += 1
            validationBoundary.exactIndex = dataRowCount
          }
        } else {
          if (millis <= previousTimestamp) {
            timestampValidationError ??= "Timestamp split mode requires unique, strictly chronological dataset timestamps"
          }
          recordBoundaryMatch(trainBoundary, timestamp, millis, dataRowCount)
          recordBoundaryMatch(validationBoundary, timestamp, millis, dataRowCount)
          previousTimestamp = millis
        }
        dataRowCount += 1
      }
    }
    if (readingHeader && !rowHasNonWhitespace) header = []
    columnIndex = 0
    field = ""
    rowTimeValue = ""
    fieldStarted = false
    fieldHasNonWhitespace = false
    rowHasNonWhitespace = false
    rowHasFields = false
  }

  for await (const chunk of stream) {
    for (const character of chunk as string) {
      if (skipLf) {
        skipLf = false
        if (character === "\n") continue
      }
      if (quoted) {
        if (quotePending) {
          if (character === '"') {
            append('"')
            quotePending = false
            continue
          }
          quoted = false
          quotePending = false
        } else if (character === '"') {
          quotePending = true
          continue
        } else {
          append(character)
          continue
        }
      }
      if (character === '"' && !fieldStarted) {
        quoted = true
        fieldStarted = true
      } else if (character === ",") {
        finishField()
      } else if (character === "\r" || character === "\n") {
        if (character === "\r") skipLf = true
        finishRow()
      } else {
        append(character)
      }
    }
  }

  if (quoted && !quotePending) throw new Error("Dataset CSV contains an unterminated quoted field")
  if (fieldStarted || field.length > 0 || fieldHasNonWhitespace || rowHasFields || columnIndex > 0) finishRow()
  if (readingHeader || timeIndex < 0 || dataRowCount === 0) {
    throw new Error("Dataset CSV must contain a header and at least one data row")
  }
  return { dataRowCount, timestampValidationError }
}

async function resolveTimestampSplit(
  input: ForecastSplitInput,
  datasetPath: string,
  timeColumn: string,
  directory: string,
): Promise<FrozenForecastSplit> {
  if (input.train_ratio !== undefined || input.val_ratio !== undefined || input.test_ratio !== undefined) {
    throw new Error("Timestamp split mode accepts only two timestamps; do not also provide split ratios")
  }
  if (input.train_end_timestamp === undefined || input.validation_end_timestamp === undefined) {
    throw new Error("Timestamp split mode requires train_end_timestamp and validation_end_timestamp")
  }

  const resolvedDatasetPath = path.isAbsolute(datasetPath) ? datasetPath : path.resolve(directory, datasetPath)
  const trainBoundary = createBoundaryMatch(input.train_end_timestamp, "train_end_timestamp")
  const validationBoundary = createBoundaryMatch(input.validation_end_timestamp, "validation_end_timestamp")
  const scan = await scanTimestampColumn(
    resolvedDatasetPath,
    timeColumn,
    trainBoundary,
    validationBoundary,
  )
  const trainBoundaryIndex = resolveBoundaryIndex(trainBoundary)
  const validationBoundaryIndex = resolveBoundaryIndex(validationBoundary)
  if (scan.timestampValidationError) throw new Error(scan.timestampValidationError)
  const dataRowCount = scan.dataRowCount
  const trainEnd = trainBoundaryIndex + 1
  const validationEnd = validationBoundaryIndex + 1
  if (trainEnd >= validationEnd) {
    throw new Error("train_end_timestamp must occur before validation_end_timestamp")
  }
  if (validationEnd >= dataRowCount) {
    throw new Error("validation_end_timestamp must leave at least one later row for the test split")
  }

  return {
    split_mode: "timestamp",
    train_ratio: trainEnd / dataRowCount,
    val_ratio: (validationEnd - trainEnd) / dataRowCount,
    test_ratio: (dataRowCount - validationEnd) / dataRowCount,
    train_end_timestamp: input.train_end_timestamp.trim(),
    validation_end_timestamp: input.validation_end_timestamp.trim(),
  }
}

export async function freezeForecastSplit(
  input: ForecastSplitInput,
  datasetPath: string,
  timeColumn: string,
  directory: string,
): Promise<FrozenForecastSplit> {
  const mode = splitMode(input)
  if (mode === "ratio") return resolveRatioSplit(input)
  return resolveTimestampSplit(input, datasetPath, timeColumn, directory)
}
