import type { Config, Message } from "./contracts"
import { AppError } from "./contracts"

export const OUTPUT_TOKEN_MAX=32000

/** Reference model defaults; transport stays OpenAI-compatible. */
export function referenceSampling(model:string) {
  const id=model.toLowerCase()
  let temperature:number|undefined,top_p:number|undefined,top_k:number|undefined
  if(id.includes("qwen"))temperature=.55
  else if(id.includes("claude"))temperature=undefined
  else if(["gemini","glm-4.6","glm-4.7","minimax-m2"].some(s=>id.includes(s)))temperature=1
  else if(id.includes("kimi-k2"))temperature=["thinking","k2.","k2p","k2-5"].some(s=>id.includes(s))?1:.6
  if(id.includes("qwen"))top_p=1
  else if(["minimax-m2","gemini","kimi-k2.5","kimi-k2p5","kimi-k2-5"].some(s=>id.includes(s)))top_p=.95
  if(id.includes("minimax-m2"))top_k=["m2.","m25","m21"].some(s=>id.includes(s))?40:20
  else if(id.includes("gemini"))top_k=64
  return {temperature,top_p,top_k}
}

export function resolveLLMPolicy(config:Config,agent?:string,modelOverride?:string,maxOutputTokens?:number) {
  const llm=config.llm
  if(!llm?.model || !llm.base_url)throw new AppError("LLM_CONFIG_MISSING","LLM model or endpoint not configured",2)
  const role=config.agents[agent??""]
  const model=modelOverride??role?.model??llm.model
  const caps=llm.model_capabilities[model]??{}
  const defaults=referenceSampling(model)
  const output=Math.min(maxOutputTokens??role?.max_output_tokens??llm.max_output_tokens,caps.max_output_tokens??OUTPUT_TOKEN_MAX)
  const context=Math.min(caps.context_window_tokens??llm.context_window_tokens,llm.context_window_tokens)
  const inputLimit=caps.input_limit_tokens
  const reserved=config.context.reserved_tokens??Math.min(20000,output)
  const usable=inputLimit?Math.min(inputLimit-reserved,context-output):context-output
  if(usable<512)throw new AppError("LLM_CONFIG_INVALID","Model limits leave insufficient input context",2)
  return {model,output,context,inputLimit,reserved,usable,supportsTools:caps.supports_tools!==false,
    temperature:caps.supports_temperature===false?undefined:role?.temperature??llm.temperature??defaults.temperature,
    top_p:role?.top_p??llm.top_p??defaults.top_p,
    top_k:caps.supports_top_k?role?.top_k??llm.top_k??defaults.top_k:undefined,
    seed:role?.seed??llm.seed,reasoning_effort:role?.reasoning_effort??llm.reasoning_effort}
}

/** Internal audit metadata never becomes provider input. */
export function modelMessages(messages:Message[]):Message[] {
  return messages.map(({role,content,tool_calls,tool_call_id,reasoning_content})=>({role,content,
    ...(tool_calls?{tool_calls}:{}),...(tool_call_id?{tool_call_id}:{}),
    ...(reasoning_content!==undefined?{reasoning_content}:{})}))
}
