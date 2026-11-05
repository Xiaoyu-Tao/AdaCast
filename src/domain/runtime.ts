/** Framework-free bridge for the independently owned forecasting domain. */
import { AsyncLocalStorage } from "node:async_hooks"
import path from "node:path"
import { readFile } from "node:fs/promises"
import { z } from "zod"
import { AppError } from "../contracts"
import type { Config, Message } from "../contracts"
import type { Store } from "../storage"
import type { LLMClient } from "../llm"
import { atomicWrite, exists, id } from "../util"
import { compute } from "../python"
import windowPrompt from "../agents/window-reflection.txt"

export type DomainScope={config:Config;store:Store;llm:LLMClient;signal:AbortSignal;agent?:string;onFatal?:(error:AppError)=>void}
const scopes=new AsyncLocalStorage<DomainScope>()
export function domainScope(){const scope=scopes.getStore();if(!scope) throw new Error("Forecast domain requires an execution scope");return scope}
export function withDomain<T>(scope:DomainScope,fn:()=>T):T{return scopes.run(scope,fn)}
export const Instance={get directory(){return path.dirname(domainScope().store.root)}}
export const Filesystem={exists,async readJson<T>(file:string):Promise<T>{return JSON.parse(await readFile(file,"utf8"))},
  async writeJson(file:string,value:unknown){await atomicWrite(file,JSON.stringify(value,null,2))},
  async findUp(name:string,from:string){let dir=path.resolve(from);for(;;){const file=path.join(dir,name);if(await exists(file))return file;const next=path.dirname(dir);if(next===dir)return;dir=next}}}
export function configuredWindowConcurrency(){return scopes.getStore()?.config.execution.window_llm_concurrency}
export function configuredWindowLlmTimeoutMs(){const scope=scopes.getStore();return scope?scope.config.execution.llm_timeout_seconds*1000:undefined}
export async function executeEngineModule(module:string,args:string[],timeoutMs?:number){const scope=domainScope();const config=timeoutMs?{...scope.config,execution:{...scope.config.execution,tool_timeout_seconds:Math.min(scope.config.execution.tool_timeout_seconds,Math.ceil(timeoutMs/1000))}}:scope.config;return compute(config,{operation:"engine_module",module,args},scope.store.root,scope.signal)}
export type ProviderID=string
export type ModelID=string
export type SessionID=string
export const ProviderID=(value:string)=>value
export const ModelID=(value:string)=>value
export const SessionID=(value:string)=>value
export const MessageID={ascending:()=>id("message")}
export const ProviderTransform={get OUTPUT_TOKEN_MAX(){
  const scope=scopes.getStore(),llm=scope?.config.llm
  const model=scope?.config.agents[scope.agent??"critic"]?.model??llm?.model
  return model?llm?.model_capabilities[model]?.max_output_tokens??32000:32000
}}
export const Permission={fromConfig:(value:Record<string,string>)=>value}
export namespace Tool {
  export type Context={sessionID:string;messageID:string;agent:string;abort:AbortSignal;callID?:string;messages:unknown[];metadata:(value:unknown)=>void;ask:(value:unknown)=>Promise<void>}
  export type Info<P extends z.ZodType=z.ZodType>={id:string;definition?:unknown;init:()=>Promise<{description:string;parameters:P;execute:(params:z.infer<P>,ctx:Context)=>Promise<{title:string;metadata:Record<string,unknown>;output:string}>}>}
  export function define<P extends z.ZodType>(name:string,definition:{description:string;parameters:P;execute:(params:z.infer<P>,ctx:Context)=>Promise<{title:string;metadata:Record<string,unknown>;output:string}>}):Info<P>{return {id:name,definition,init:async()=>definition} as Info<P>}
}
export namespace MessageV2 {
  export const Format=z.object({type:z.literal("json_schema"),schema:z.record(z.string(),z.unknown()),retryCount:z.number().optional()})
  export type WithParts={info:any;parts:any[]}
  export async function get(_input:unknown):Promise<WithParts>{const scope=domainScope();return {info:{role:"assistant",providerID:scope.config.llm?.provider,modelID:scope.config.agents[scope.agent??"critic"]?.model??scope.config.llm?.model},parts:[]}}
}
const children=new Map<string,AbortController>()
export const Session={async create(input:{parentID:string;title:string;permission:unknown}){const key=id("window");children.set(key,new AbortController());domainScope().store.set("domain-session:"+key,{...input,created_at:new Date().toISOString()});return {id:key}}}
export const SessionPrompt={
  async cancel(sessionID:string){children.get(sessionID)?.abort();children.delete(sessionID)},
  async prompt(input:{sessionID:string;messageID:string;model:{providerID:string;modelID:string};agent:string;system:string;tools:Record<string,unknown>;maxOutputTokens?:number;format?:z.infer<typeof MessageV2.Format>;parts:{type:string;text:string}[]}):Promise<MessageV2.WithParts>{
    const scope=domainScope(),child=children.get(input.sessionID)
    const signal=child?AbortSignal.any([scope.signal,child.signal]):scope.signal
    const messages:Message[]=[{role:"system",content:windowPrompt+"\n\n"+input.system},{role:"user",content:input.parts.map(p=>p.text).join("\n\n")}]
    const tools=input.format?[{type:"function",function:{name:"StructuredOutput",description:"Submit the required structured response exactly once.",parameters:input.format.schema}}]:[]
    scope.store.append(input.sessionID,messages[1])
    try {
      const response=await scope.llm.complete(messages,tools,"window:"+input.sessionID,signal,input.model.modelID,{maxOutputTokens:input.maxOutputTokens,toolChoice:input.format?{type:"function",function:{name:"StructuredOutput"}}:undefined,agent:input.agent})
      scope.store.append(input.sessionID,response)
      const parts:any[]=[]
      if(response.reasoning_content)parts.push({type:"reasoning",text:response.reasoning_content})
      if(response.content)parts.push({type:"text",text:response.content})
      let structured:unknown
      for(const call of response.tool_calls??[]){
        if(call.function.name!=="StructuredOutput" || !input.format)throw new Error("Isolated window call attempted an unavailable tool")
        structured=JSON.parse(call.function.arguments)
        parts.push({type:"tool",tool:"StructuredOutput",callID:call.id,state:{status:"completed",input:structured,output:JSON.stringify(structured)}})
        scope.store.append(input.sessionID,{role:"tool",tool_call_id:call.id,content:"Structured output submitted"})
      }
      return {info:{role:"assistant",providerID:input.model.providerID,modelID:input.model.modelID,structured},parts}
    }catch(error){
      // Local child cancellation must not poison sibling requests or mask the
      // original controller error. Parent cancellation already reaches all children.
      if(error instanceof AppError && (error.exitCode===5 || scope.signal.aborted))scope.onFatal?.(error)
      throw error
    }finally{children.delete(input.sessionID)}
  }
}
