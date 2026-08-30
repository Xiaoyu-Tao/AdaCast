export * from "./client.js"
export * from "./server.js"

import { createAdaCastClient } from "./client.js"
import { createAdaCastServer } from "./server.js"
import type { ServerOptions } from "./server.js"

export async function createAdaCast(options?: ServerOptions) {
  const server = await createAdaCastServer({
    ...options,
  })

  const client = createAdaCastClient({
    baseUrl: server.url,
  })

  return {
    client,
    server,
  }
}
