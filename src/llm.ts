import { z } from "zod"
import path from "node:path"
import { AppError, type Config, type Message, type TokenUsage } from "./contracts"
import type { Store } from "./storage"
import { checkAbort, errorText, sanitize, writeJSON } from "./util"
import { estimateTokens } from "./context"
import { modelMessages, resolveLLMPolicy } from "./llm-policy"

const UsageSchema=z.object({total_tokens:z.number().int().nonnegative().optional(),prompt_tokens:z.number().int().nonnegative().optional(),completion_tokens:z.number().int().nonnegative().optional(),
  prompt_tokens_details:z.object({cached_tokens:z.number().nonnegative().optional()}).passthrough().optional(),
  completion_tokens_details:z.object({reasoning_tokens:z.number().nonnegative().optional()}).passthrough().optional()}).passthrough()
const ResponseSchema=z.object({choices:z.array(z.object({finish_reason:z.string().nullable(),message:z.object({role:z.literal("assistant"),content:z.string().nullable().optional(),reasoning_content:z.string().nullable().optional(),
  tool_calls:z.array(z.object({id:z.string().min(1),type:z.literal("function"),function:z.object({name:z.string(),arguments:z.string()})})).optional()})})).min(1),usage:UsageSchema.optional()})
export type CompletionOptions={maxOutputTokens?:number;toolChoice?:unknown;agent?:string}

export class LLMClient {
  constructor(public config:Config,public store:Store){}
  async complete(messages:Message[],tools:unknown[],purpose:string,signal:AbortSignal,model?:string,options:CompletionOptions={}):Promise<Message>{
    const llm=this.config.llm!
    const policy=resolveLLMPolicy(this.config,options.agent,model,options.maxOutputTokens)
    if(tools.length&&!policy.supportsTools)throw new AppError("LLM_TOOLS_UNSUPPORTED","Selected model does not support the task's tools",2)
    const available=new Set((tools as {function?:{name?:string}}[]).map(t=>t.function?.name).filter(Boolean))
    for(let attempt=0;attempt<=llm.max_retries;attempt++){
      checkAbort(signal)
      const body={model:policy.model,messages:modelMessages(messages),
        ...(policy.temperature!==undefined?{temperature:policy.temperature}:{}),...(policy.top_p!==undefined?{top_p:policy.top_p}:{}),
        ...(policy.top_k!==undefined?{top_k:policy.top_k}:{}),...(policy.seed!==undefined?{seed:policy.seed}:{}),
        ...(policy.reasoning_effort!==undefined?{reasoning_effort:policy.reasoning_effort}:{}),
        max_tokens:policy.output,stream:false,...(tools.length?{tools,tool_choice:options.toolChoice??"auto"}:{})}
      const inputTokens=estimateTokens(body)
      if(inputTokens>policy.usable)throw new AppError("CONTEXT_TOO_LARGE","Serialized LLM request exceeds configured model input allowance",3)
      const callId=this.store.reserve(inputTokens+policy.output,this.config)
      const auditDir=path.join(this.store.root,"runtime/llm-calls",callId)
      await writeJSON(path.join(auditDir,"request.json"),{call_id:callId,purpose,attempt,request:body})
      const timeout=AbortSignal.timeout(this.config.execution.llm_timeout_seconds*1000)
      const combined=AbortSignal.any([signal,timeout])
      let retryable=false
      try{
        const response=await fetch(llm.base_url!.replace(/\/$/,"")+"/chat/completions",{method:"POST",signal:combined,
          headers:{"Content-Type":"application/json",Authorization:`Bearer ${process.env[llm.api_key_env]}`},body:JSON.stringify(body)})
        const text=await response.text()
        await writeJSON(path.join(auditDir,"response.json"),{status:response.status,body:sanitize(text)})
        if(!response.ok){
          if(response.status===413||/context_length_exceeded|maximum context length|context window|too many tokens|prompt is too long/i.test(text))throw new AppError("LLM_CONTEXT_OVERFLOW","Provider rejected the conversation context size",4)
          retryable=response.status===429||response.status>=500
          throw new AppError("LLM_HTTP_ERROR",`LLM returned HTTP ${response.status}: ${sanitize(text).slice(0,800)}`,4,retryable)
        }
        const parsed=ResponseSchema.parse(JSON.parse(text))
        const usage:TokenUsage|undefined=parsed.usage?{input:Math.max(0,(parsed.usage.prompt_tokens??0)-(parsed.usage.prompt_tokens_details?.cached_tokens??0)),output:parsed.usage.completion_tokens,
          cache_read:parsed.usage.prompt_tokens_details?.cached_tokens,reasoning:parsed.usage.completion_tokens_details?.reasoning_tokens,
          total:parsed.usage.total_tokens??((parsed.usage.prompt_tokens??0)+(parsed.usage.completion_tokens??0)||undefined)}:undefined
        this.store.settle(callId,"received",usage?.total)
        const choice=parsed.choices[0],message=choice.message
        if(choice.finish_reason==="length")throw new AppError("LLM_TRUNCATED","Model response exceeded output limit",4)
        if(!message.content?.trim()&&!message.tool_calls?.length&&!message.reasoning_content?.trim())throw new AppError("LLM_EMPTY","Model returned no content or tool calls",4)
        if(message.tool_calls&&new Set(message.tool_calls.map(c=>c.id)).size!==message.tool_calls.length)throw new AppError("LLM_PROTOCOL","Duplicate tool call identifiers",4)
        for(const call of message.tool_calls??[]){
          const lower=call.function.name.toLowerCase()
          if(!available.has(call.function.name)&&available.has(lower)){
            this.store.event("llm.tool_name_repaired",{callId,from:call.function.name,to:lower});call.function.name=lower
          }
        }
        await writeJSON(path.join(auditDir,"parsed.json"),JSON.parse(sanitize(JSON.stringify(message))))
        this.store.event("llm.completed",{callId,purpose,attempt})
        return {...message,content:message.content??null,audit:{usage,model:policy.model,finish_reason:choice.finish_reason}}
      }catch(error){
        this.store.settle(callId,"failed")
        if(error instanceof AppError)retryable=error.retryable
        else if(error instanceof TypeError || timeout.aborted || error instanceof Error && (
          error.name==="TimeoutError" || ["ETIMEDOUT","ECONNRESET","ECONNREFUSED","EPIPE"].includes(String((error as Error&{code?:string}).code))
        ))retryable=true
        await writeJSON(path.join(auditDir,"error.json"),{error:sanitize(errorText(error)),retryable})
        if(signal.aborted)checkAbort(signal)
        if(!retryable||attempt>=llm.max_retries)throw error instanceof AppError?error:new AppError("LLM_FAILED",sanitize(errorText(error)),4,retryable)
        await Bun.sleep(Math.min(1000*2**attempt,4000))
      }
    }
    throw new AppError("LLM_FAILED","Retry budget exhausted")
  }
  async structured<T>(schema:z.ZodType<T>,system:string,input:unknown,purpose:string,signal:AbortSignal):Promise<T>{
    const jsonSchema=z.toJSONSchema(schema)
    const messages:Message[]=[{role:"system",content:system+"\nReturn JSON only matching this schema:\n"+JSON.stringify(jsonSchema)},{role:"user",content:JSON.stringify(input)}]
    for(let attempt=0;attempt<3;attempt++){
      const message=await this.complete(messages,[],purpose,signal)
      if(message.tool_calls?.length)throw new AppError("ISOLATION_VIOLATION","Isolated call attempted tool use",3)
      try{return schema.parse(JSON.parse((message.content??"").trim().replace(/^```(?:json)?\s*/,"").replace(/\s*```$/,"")))}
      catch(error){if(attempt===2)throw new AppError("LLM_SCHEMA_INVALID",sanitize(errorText(error)),4,true);messages.push(message,{role:"user",content:"Invalid JSON/schema. Correct the format only. Error: "+errorText(error).slice(0,1000)})}
    }
    throw new AppError("LLM_SCHEMA_INVALID","Invalid structured response")
  }
}
