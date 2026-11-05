import path from "node:path"
import { mkdir, copyFile, chmod, readdir, rm, cp } from "node:fs/promises"
const root=path.resolve(import.meta.dir,"..")
await mkdir(path.join(root,"dist"),{recursive:true})
await mkdir(path.join(root,"dist/python"),{recursive:true})
for(const file of await readdir(path.join(root,"dist/python")))if(/^adacast_lite_ml-.*\.whl$/.test(file))await rm(path.join(root,"dist/python",file))
for(const args of [
  [process.execPath,"build","src/cli.ts","--compile","--outfile","dist/adacast-lite"],
  ["uv","build","--wheel","--project","python","--out-dir",path.join(root,"dist/python")]
]) {
  const process_=Bun.spawn(args,{cwd:root,stdout:"inherit",stderr:"inherit"})
  if(await process_.exited!==0) throw new Error("Build failed: "+args[0])
}
await copyFile(path.join(root,"python/model-overrides.txt"),path.join(root,"dist/python/model-overrides.txt"))
await copyFile(path.join(root,"scripts/install.sh"),path.join(root,"dist/install.sh"))
await chmod(path.join(root,"dist/install.sh"),0o755)
await copyFile(path.join(root,"README.md"),path.join(root,"dist/README.md"))
await copyFile(path.join(root,"src/domain/LICENSE"),path.join(root,"dist/ADACAST_ENGINE_LICENSE.txt"))
for(const file of ["LICENSE","THIRD_PARTY_NOTICES.md"])
  await copyFile(path.join(root,file),path.join(root,"dist",file))
await mkdir(path.join(root,"dist/docs"),{recursive:true})
await copyFile(path.join(root,"docs/ALIGNMENT_0_3.md"),path.join(root,"dist/docs/ALIGNMENT_0_3.md"))
await copyFile(path.join(root,"docs/BEHAVIOR_ALIGNMENT.md"),path.join(root,"dist/docs/BEHAVIOR_ALIGNMENT.md"))
await cp(path.join(root,"docs/parity-results/scoped-0.3"),path.join(root,"dist/docs/parity-results/scoped-0.3"),{recursive:true})
await cp(path.join(root,"docs/parity-results/aligned"),path.join(root,"dist/docs/parity-results/aligned"),{recursive:true})
console.log("Release artifacts: "+path.join(root,"dist"))
