import { defineConfig } from "drizzle-kit"
import { xdgData } from "xdg-basedir"
import path from "path"

const dataDirectory = path.join(xdgData!, "adacast")
const configuredDatabase = process.env.ADACAST_DB
const databaseUrl = !configuredDatabase
  ? path.join(dataDirectory, "adacast.db")
  : configuredDatabase === ":memory:" || path.isAbsolute(configuredDatabase)
    ? configuredDatabase
    : path.join(dataDirectory, configuredDatabase)

export default defineConfig({
  dialect: "sqlite",
  schema: "./src/**/*.sql.ts",
  out: "./migration",
  dbCredentials: {
    url: databaseUrl,
  },
})
