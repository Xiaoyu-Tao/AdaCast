import ts from "typescript"
import path from "node:path"
import { readFile, readdir } from "node:fs/promises"
const root=path.resolve(import.meta.dir,"..")
const errors:string[]=[]
const manifest=JSON.parse(await readFile(path.join(root,"package.json"),"utf8"))
const dependencies={...manifest.dependencies,...manifest.devDependencies,...manifest.optionalDependencies}
const contained=(file:string)=>{const relative=path.relative(root,file);return relative==="" || (!relative.startsWith(".."+path.sep)&&relative!==".."&&!path.isAbsolute(relative))}
if(manifest.workspaces) errors.push("The project must own its dependency root")
for(const [name,version] of Object.entries(dependencies)) {
  if(typeof version!=="string" || /^(workspace:|link:)/.test(version)) errors.push("External workspace dependency: "+name)
  if(typeof version==="string"&&version.startsWith("file:")&&!contained(path.resolve(root,version.slice(5)))) errors.push("External local dependency: "+name)
}
async function scan(dir:string) {
  for(const entry of await readdir(dir,{withFileTypes:true})) {
    if(["node_modules","dist",".venv","__pycache__","workspaces"].includes(entry.name)) continue
    const file=path.join(dir,entry.name)
    if(entry.isSymbolicLink()) {errors.push("Symlink: "+path.relative(root,file));continue}
    if(entry.isDirectory()) await scan(file)
    else if(/\.(ts|py|json|toml|sh)$/.test(entry.name) && file!==import.meta.path) {
      const text=await readFile(file,"utf8")
      if(entry.name.endsWith(".py") && /(?:^|\n)\s*(?:from|import)\s+adacast_ml(?:[.\s]|$)/.test(text))errors.push("Original Python import in "+path.relative(root,file))
      if(entry.name.endsWith(".ts")) {
        const source=ts.createSourceFile(file,text,ts.ScriptTarget.Latest,true)
        const imports:string[]=[]
        const visit=(node:ts.Node)=>{
          if((ts.isImportDeclaration(node)||ts.isExportDeclaration(node))&&node.moduleSpecifier&&ts.isStringLiteral(node.moduleSpecifier))imports.push(node.moduleSpecifier.text)
          if(ts.isCallExpression(node)&&node.expression.kind===ts.SyntaxKind.ImportKeyword&&node.arguments[0]&&ts.isStringLiteral(node.arguments[0]))imports.push(node.arguments[0].text)
          ts.forEachChild(node,visit)
        }
        visit(source)
        for(const specifier of imports) {
          if(specifier.startsWith(".")) {
            if(!contained(path.resolve(path.dirname(file),specifier)))errors.push("Import escapes project: "+path.relative(root,file))
          }else if(!specifier.startsWith("node:")&&!specifier.startsWith("bun:")) {
            const dependency=specifier.startsWith("@")?specifier.split("/").slice(0,2).join("/"):specifier.split("/")[0]
            if(!Object.hasOwn(dependencies,dependency))errors.push("Undeclared dependency "+dependency+" in "+path.relative(root,file))
          }
        }
      }
    }
  }
}
await scan(root)
if(errors.length) throw new Error("External project coupling found: "+errors.join(", "))
console.log("Independent source/dependency check passed")
