export const domain = (() => {
  if ($app.stage === "production") return "adacast.example.invalid"
  if ($app.stage === "dev") return "dev.adacast.example.invalid"
  return `${$app.stage}.dev.adacast.example.invalid`
})()

export const zoneID = "430ba34c138cfb5360826c4909f99be8"

new cloudflare.RegionalHostname("RegionalHostname", {
  hostname: domain,
  regionKey: "us",
  zoneId: zoneID,
})
