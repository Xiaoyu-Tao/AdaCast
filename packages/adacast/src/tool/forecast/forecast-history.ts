type AttemptRecord = {
  run_id: string
  attempt_index?: number
}

export function experimentAttemptIndex(record: AttemptRecord): number {
  return record.attempt_index ?? 1
}

export function deduplicateExperimentsByRunId<T extends AttemptRecord>(records: T[]): T[] {
  const unique: T[] = []
  const indexByRunId = new Map<string, number>()

  for (const record of records) {
    const existingIndex = indexByRunId.get(record.run_id)
    if (existingIndex === undefined) {
      indexByRunId.set(record.run_id, unique.length)
      unique.push(record)
      continue
    }
    if (experimentAttemptIndex(record) > experimentAttemptIndex(unique[existingIndex])) {
      unique[existingIndex] = record
    }
  }

  return unique
}
