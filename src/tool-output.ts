/** Reference truncation limits; full bytes stay available through read_tool_output. */
export function truncateToolOutput(text:string,maxLines=2000,maxBytes=50*1024){
  const lines=text.split("\n")
  if(lines.length<=maxLines&&Buffer.byteLength(text)<=maxBytes)return {truncated:false,preview:text}
  const output:string[]=[];let bytes=0
  for(const line of lines){const size=Buffer.byteLength(line)+(output.length?1:0);if(output.length>=maxLines||bytes+size>maxBytes)break;output.push(line);bytes+=size}
  return {truncated:true,preview:output.join("\n")}
}
