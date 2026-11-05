import { z } from "zod"
import { ConfigSchema } from "../src/contracts"
import { writeJSON } from "../src/util"
await writeJSON(new URL("../schemas/task.schema.json",import.meta.url).pathname,z.toJSONSchema(ConfigSchema,{io:"input",target:"draft-7"}))
