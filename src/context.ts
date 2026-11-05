import path from "node:path"
import compactionPrompt from "./agents/compaction.txt"
import { AppError, type Message } from "./contracts"
import { canonical, digest, writeJSON } from "./util"
import { modelMessages } from "./llm-policy"
import type { Store } from "./storage"

export const estimateTokens=(value:unknown)=>Math.max(0,Math.round((typeof value==="string"?value:JSON.stringify(value)).length/4))
export const PRUNE_MINIMUM=20000,PRUNE_PROTECT=40000,COMPACTION_BUFFER=20000
const CLEARED="[Old tool result content cleared]"
const PROTECTED_TOOLS=new Set(["skill","forecast_skill","list_skills","import_memory"])
export const CONTINUATION_PROMPT=`Provide a detailed prompt for continuing our conversation above.
Focus on information that would be helpful for continuing the conversation, including what we did, what we're doing, which files we're working on, and what we're going to do next.
The summary that you construct will be used so that another agent can read it and continue the work.

When constructing the summary, try to stick to this template:
---
## Goal

[What goal(s) is the user trying to accomplish?]

## Instructions

- [What important instructions did the user give you that are relevant]
- [If there is a plan or spec, include information about it so next agent can continue using it]

## Discoveries

[What notable things were learned during this conversation that would be useful for the next agent to know when continuing the work]

## Accomplished

[What work has been completed, what work is still in progress, and what is left?]

## Relevant files / directories

[Construct a structured list of relevant files that have been read, edited, or created that pertain to the task at hand.]
---`

export function messageGroups(messages:Message[]) {
  const groups:Message[][]=[]
  for(let i=0;i<messages.length;i++){
    const msg=messages[i]
    if(msg.role==="tool")throw new AppError("MESSAGE_PROTOCOL","Orphan tool result")
    const group=[msg]
    if(msg.tool_calls?.length){
      const pending=new Set(msg.tool_calls.map(c=>c.id))
      if(pending.size!==msg.tool_calls.length)throw new AppError("MESSAGE_PROTOCOL","Duplicate tool call identifier")
      while(i+1<messages.length&&messages[i+1].role==="tool"){
        const result=messages[++i]
        if(!result.tool_call_id||!pending.delete(result.tool_call_id))throw new AppError("MESSAGE_PROTOCOL","Duplicate or unmatched tool result")
        group.push(result)
      }
      if(pending.size)throw new AppError("MESSAGE_PROTOCOL","Pending tools must be completed before calling the LLM")
    }
    groups.push(group)
  }
  return groups
}

type Checkpoint={through:number;prefix_hash:string;summary:string;hash:string;observed_length:number}
export type ContextOptions={auto?:boolean;prune?:boolean;reserved_tokens?:number;input_limit_tokens?:number;
  prune_minimum_tokens?:number;prune_protect_tokens?:number;force?:boolean;model?:string;compaction_input_tokens?:number;compaction_model?:string}

export function usableContext(window:number,output:number,options:ContextOptions={}){
  const reserved=options.reserved_tokens??Math.min(COMPACTION_BUFFER,output)
  return options.input_limit_tokens?Math.min(options.input_limit_tokens-reserved,window-output):window-output
}

function usageCount(message?:Message){
  const usage=message?.audit?.usage
  if(!usage)return 0
  return usage.total||(usage.input??0)+(usage.output??0)+(usage.cache_read??0)+(usage.cache_write??0)
}

/** Project pruning into provider input; immutable original messages remain in SQLite. */
function pruneHistory(store:Store,session:string,messages:Message[],start:number,options:ContextOptions){
  const key="context-pruned:"+session
  const previous=store.get<number[]>(key)??[]
  const pruned=new Set(previous)
  if(options.prune!==false){
    let turns=0,total=0,removed=0
    const candidates:number[]=[]
    for(let i=messages.length-1;i>=start;i--){
      const message=messages[i]
      if(message.role==="user")turns++
      if(turns<2)continue
      if(message.role!=="tool")continue
      if(pruned.has(i))break
      const call=messages.slice(start,i).findLast(m=>m.tool_calls?.some(c=>c.id===message.tool_call_id))?.tool_calls?.find(c=>c.id===message.tool_call_id)
      if(call&&PROTECTED_TOOLS.has(call.function.name))continue
      const count=estimateTokens(message.content??"")
      total+=count
      if(total>(options.prune_protect_tokens??PRUNE_PROTECT)){removed+=count;candidates.push(i)}
    }
    if(removed>(options.prune_minimum_tokens??PRUNE_MINIMUM)){
      for(const index of candidates)pruned.add(index)
      store.set(key,[...pruned].sort((a,b)=>a-b))
      store.event("context.pruned",{session,messages:candidates.length,tokens:removed})
    }
  }
  return messages.slice(start).map((message,index)=>pruned.has(start+index)?{...message,content:CLEARED}:message)
}

export async function buildContext(store:Store,session:string,system:string,tools:unknown,window:number,output:number,
  compact?:(messages:Message[])=>Promise<string>,options:ContextOptions={}):Promise<Message[]> {
  const raw=store.messages(session)
  messageGroups(raw)
  const key="context-checkpoint:"+session
  const checkpoint=store.get<Checkpoint>(key)
  if(checkpoint&&(checkpoint.through>raw.length||checkpoint.prefix_hash!==digest(canonical(raw.slice(0,checkpoint.through)))))throw new AppError("CONTEXT_INTEGRITY","Compaction checkpoint no longer matches its message history",3)
  const start=checkpoint?.through??0
  const projected=pruneHistory(store,session,raw,start,options)
  const continuation:Message={role:"user",content:"Continue the pending task using the frozen constraints and committed tool results. Read current task state when needed."}
  const history:Message[]=[...(checkpoint?[{role:"assistant" as const,content:checkpoint.summary},continuation]:[]),...projected]
  const prefix:Message={role:"system",content:system}
  const usable=usableContext(window,output,options)
  const tokens=estimateTokens({messages:modelMessages([prefix,...history]),tools})
  const last=raw.findLast(m=>m.role==="assistant")
  const overflow=options.force||tokens>=usable||(raw.length>(checkpoint?.observed_length??0)&&usageCount(last)>=usable)
  if(!overflow)return modelMessages([prefix,...history])
  if(options.auto===false&&!options.force){if(tokens>=usable)throw new AppError("CONTEXT_TOO_LARGE","Automatic context compaction is disabled",3);return modelMessages([prefix,...history])}
  if(!compact)throw new AppError("CONTEXT_COMPACTION_UNAVAILABLE","A semantic compactor is required to continue this session",3)
  const groups=messageGroups(projected)
  // Preserve the complete most recent user turn on provider overflow when possible.
  let tailStart=projected.findLastIndex(m=>m.role==="user")
  if(tailStart<=0)tailStart=projected.length-(groups.at(-1)?.length??0)
  if(tailStart<=0)throw new AppError("CONTEXT_TOO_LARGE","A single complete message group exceeds the model input allowance",3)
  const dropped=projected.slice(0,tailStart),tail=projected.slice(tailStart)
  const through=start+tailStart
  const source:Message[]=[{role:"system",content:compactionPrompt},
    ...(checkpoint?[{role:"assistant" as const,content:checkpoint.summary},continuation]:[]),...modelMessages(dropped),
    {role:"user",content:CONTINUATION_PROMPT+"\n\nDurable task state:\n"+JSON.stringify(store.get("handoff")??{})}]
  const compactionBudget=Math.min(usable,options.compaction_input_tokens??usable)
  const hash=digest(canonical({source,through,model:options.model,compaction_model:options.compaction_model,compactionBudget,window,output}))
  let summary=store.get<string>("context-summary:"+hash)
  if(!summary){
    const header=source[0],instruction=source.at(-1)!
    const pending=messageGroups(source.slice(1,-1))
    let carry="",batch:Message[]=[]
    const flush=async()=>{
      if(!batch.length)return
      const input=[header,...(carry?[{role:"assistant" as const,content:carry}]:[]),...batch,instruction]
      if(estimateTokens(input)+128>compactionBudget)throw new AppError("CONTEXT_TOO_LARGE","A complete message group cannot fit the compaction model",3)
      const partHash=digest(canonical({input,model:options.model,window,output}))
      carry=store.get<string>("context-summary-part:"+partHash)??(await compact(input)).trim()
      if(!carry)throw new AppError("CONTEXT_COMPACTION_EMPTY","Compaction returned an empty summary",4,true)
      store.set("context-summary-part:"+partHash,carry)
      batch=[]
    }
    for(const group of pending){
      if(batch.length&&estimateTokens([header,...(carry?[{role:"assistant",content:carry}]:[]),...batch,...group,instruction])+128>compactionBudget)await flush()
      batch.push(...group)
    }
    await flush()
    summary=carry
    store.set("context-summary:"+hash,summary)
  }
  const rebuilt:Message[]=[prefix,{role:"assistant",content:summary},continuation,...tail]
  if(estimateTokens({messages:modelMessages(rebuilt),tools})>=usable)throw new AppError("CONTEXT_TOO_LARGE","Compacted conversation still exceeds model input allowance",3)
  const next:Checkpoint={through,prefix_hash:digest(canonical(raw.slice(0,through))),summary,hash,observed_length:raw.length}
  await writeJSON(path.join(store.root,"runtime/context-snapshots",hash+".json"),{input_hash:hash,summary,through,group_count:messageGroups(dropped).length,forced:!!options.force})
  store.set(key,next)
  store.event("context.compacted",{session,hash,through,groups:messageGroups(dropped).length,forced:!!options.force})
  return modelMessages(rebuilt)
}
