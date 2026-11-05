import { resolveLLMPolicy } from "./llm-policy"
import { truncateToolOutput } from "./tool-output"
import path from "node:path"
import { z } from "zod"
import { AppError, type Config, type Message, type ToolContext, type ToolDefinition } from "./contracts"
import { buildContext } from "./context"
import type { Store } from "./storage"
import type { LLMClient } from "./llm"
import { canonical, checkAbort, digest, errorText, sanitize, writeJSON } from "./util"

export class AgentRuntime {
  tools=new Map<string,ToolDefinition>()
  constructor(public config:Config,public store:Store,public llm:LLMClient,
    public system:(agent:string)=>string,public signal:AbortSignal){}
  register(tool:ToolDefinition) {if(this.tools.has(tool.name)) throw new Error("Duplicate tool: "+tool.name);this.tools.set(tool.name,tool)}
  available(agent:string) {
    const custom=this.config.agents[agent]?.tools
    return [...this.tools.values()].filter(t=>(t.roles.includes("*") || t.roles.includes(agent)) && (!custom || custom.includes(t.name)))
  }
  async executePending(session:string,agent:string,depth:number) {
    const messages=this.store.messages(session)
    let index=messages.length-1
    while(index>=0 && messages[index].role==="tool") index--
    const message=messages[index]
    if(!message?.tool_calls?.length) return
    const replied=new Set(messages.slice(index+1).map(m=>m.tool_call_id))
    for(const call of message.tool_calls) {
      if(replied.has(call.id)) continue
      checkAbort(this.signal)
      const key="tool:"+digest(session+":"+call.id)
      const hash=digest(canonical(call.function))
      const previous=this.store.operation(key)
      let result:unknown
      if(previous?.status==="completed") {
        const record=previous.value as any
        if(record.input_hash!==hash) throw new AppError("MESSAGE_PROTOCOL","Tool call id was reused for different arguments",3)
        result=record.output
      } else {
        const tool=this.available(agent).find(t=>t.name===call.function.name)
        if(previous?.status==="started" && !tool?.replaySafe)
          throw new AppError("SIDE_EFFECT_UNCERTAIN","Interrupted non-idempotent tool requires inspection: "+call.function.name,3)
        this.store.saveOperation(key,"started",{input_hash:hash})
        try {
          if(!tool) throw new AppError("TOOL_DENIED","Unknown or unavailable tool "+call.function.name,3)
          const args=tool.schema.parse(JSON.parse(call.function.arguments))
          result=await tool.execute(args,{sessionId:session,agent,depth,callId:call.id,signal:this.signal})
        } catch(e) {
          if(this.signal.aborted || (e instanceof AppError && ([5,130,143].includes(e.exitCode) || e.code==="CRASH_THRESHOLD"))) throw e
          result={error:{code:e instanceof AppError?e.code:"TOOL_INVALID",message:sanitize(errorText(e)),retryable:e instanceof AppError?e.retryable:false}}
        }
      }
      let output=JSON.stringify(result ?? null)
      const truncated=truncateToolOutput(output,this.config.context.tool_output_max_lines,this.config.context.tool_output_max_bytes)
      if(truncated.truncated) {
        const outputId=digest(output)
        await writeJSON(path.join(this.store.root,"runtime/tool-outputs",outputId+".json"),result)
        output=JSON.stringify({truncated:true,output_id:outputId,preview:truncated.preview,note:"Use read_tool_output with offsets to inspect the full result."})
      }
      this.store.transaction(()=>{
        this.store.saveOperation(key,"completed",{input_hash:hash,output:result})
        this.store.append(session,{role:"tool",tool_call_id:call.id,content:output})
        this.store.event("tool.completed",{session,agent,tool:call.function.name,callId:call.id})
      })
    }
  }
  async run(agent:string,session:string,request:string,gate?:()=>string[],depth=0,followup=false):Promise<string> {
    if(depth>this.config.execution.max_subagent_depth) throw new AppError("SUBAGENT_DEPTH","Subagent recursion limit exceeded",3)
    const requestHash=digest(request)
    if(followup && this.store.get("last-request:"+session)!==requestHash && this.store.messages(session).length) {
      await this.executePending(session,agent,depth)
      this.store.append(session,{role:"user",content:request})
      this.store.set("completed-session:"+session,null)
    }
    if(this.store.get<string>("completed-session:"+session)) return this.store.get<string>("completed-session:"+session)!
    if(!this.store.messages(session).length) this.store.append(session,{role:"user",content:request})
    this.store.set("last-request:"+session,requestHash)
    const definitions=this.available(agent).map(t=>({type:"function",function:{name:t.name,description:t.description,
      parameters:z.toJSONSchema(t.schema,{target:"draft-7"})}}))
    const policy=resolveLLMPolicy(this.config,agent)
    const compactionPolicy=resolveLLMPolicy(this.config,"compaction",this.config.agents.compaction?.model??policy.model)
    const contextOptions={...this.config.context,input_limit_tokens:policy.inputLimit,model:policy.model,compaction_input_tokens:compactionPolicy.usable,compaction_model:compactionPolicy.model}
    const compact=async(messages:Message[]) => (await this.llm.complete(this.config.agents.compaction?.prompt?[{...messages[0],content:messages[0].content+"\n\n"+this.config.agents.compaction.prompt},...messages.slice(1)]:messages,[],"compaction:"+session,this.signal,this.config.agents.compaction?.model??policy.model,{agent:"compaction"})).content??""
    let step=this.store.get<number>("steps:"+session) ?? 0
    while(step<this.config.execution.max_steps_per_agent) {
      checkAbort(this.signal)
      await this.executePending(session,agent,depth)
      const messages=await buildContext(this.store,session,this.system(agent),definitions,policy.context,policy.output,compact,contextOptions)
      this.store.set("steps:"+session,++step)
      let response:Message
      try {response=await this.llm.complete(messages,definitions,`agent:${agent}:${session}:${step}`,this.signal,policy.model,{agent})}
      catch(error){
        if(!(error instanceof AppError)||!["LLM_CONTEXT_OVERFLOW","CONTEXT_TOO_LARGE"].includes(error.code)||!this.config.context.auto)throw error
        const recovered=await buildContext(this.store,session,this.system(agent),definitions,policy.context,policy.output,compact,{...contextOptions,force:true})
        this.store.event("context.overflow_recovery",{session,agent,step})
        response=await this.llm.complete(recovered,definitions,`agent:${agent}:${session}:${step}:context-recovery`,this.signal,policy.model,{agent})
      }
      this.store.append(session,response)
      if(response.tool_calls?.length) continue
      if(!response.content?.trim()&&response.reasoning_content){this.store.append(session,{role:"user",content:"Return the required final response or use the available tools to continue the pending task."});continue}
      const missing=gate?.() ?? []
      if(missing.length) {this.store.append(session,{role:"user",content:"Required work is incomplete. Use the tools to resolve: "+missing.join("; ")});continue}
      const text=response.content ?? ""
      this.store.set("completed-session:"+session,text)
      this.store.event("agent.completed",{agent,session,steps:step});return text
    }
    throw new AppError("AGENT_STEP_LIMIT",`Agent ${agent} exceeded its step budget`,5,true)
  }
}
