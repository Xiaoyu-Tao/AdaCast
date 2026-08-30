import type { Argv } from "yargs"
import { cmd } from "./cmd"
import { formatForecastStatus, readForecastStatus } from "../../tool/forecast/status"

const ForecastStatusCommand = cmd({
  command: "status",
  describe: "show local forecast workspace status",
  builder: (yargs: Argv) => {
    return yargs.option("json", {
      describe: "print structured JSON",
      type: "boolean",
      default: false,
    })
  },
  handler: async (args) => {
    const status = await readForecastStatus(process.cwd())

    if (args.json) {
      console.log(JSON.stringify(status, null, 2))
      return
    }

    process.stdout.write(formatForecastStatus(status))
  },
})

export const ForecastCommand = cmd({
  command: "forecast",
  describe: "inspect forecast workspace state",
  builder: (yargs: Argv) => yargs.command(ForecastStatusCommand).demandCommand(),
  handler: () => {},
})
