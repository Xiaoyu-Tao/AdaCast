export function sessionLineage(session: { id: string; parentID?: string }[], sessionID: string) {
  const parent = session.reduce((acc, item) => {
    if (item.parentID) acc.set(item.id, item.parentID)
    return acc
  }, new Map<string, string>())
  const seen = new Set([sessionID])
  const ids = [sessionID]

  for (const id of ids) {
    const parentID = parent.get(id)
    if (!parentID || seen.has(parentID)) continue
    seen.add(parentID)
    ids.push(parentID)
  }

  return ids
}

export function autoAcceptsPermission(
  autoAccept: ReadonlySet<string>,
  session: { id: string; parentID?: string }[],
  permission: { sessionID: string },
) {
  return sessionLineage(session, permission.sessionID).some((id) => autoAccept.has(id))
}
