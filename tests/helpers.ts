import { mkdtemp, writeFile, rm } from "node:fs/promises"
import path from "node:path"
import os from "node:os"
import { ConfigSchema } from "../src/contracts"

export async function fixture(overrides:Record<string,any>={}) {
  const dir=await mkdtemp(path.join(os.tmpdir(),"adacast-lite-test-"))
  const data=path.join(dir,"series.csv")
  const rows=["date,target"]
  for(let i=0;i<360;i++) rows.push(new Date(Date.UTC(2025,0,1)+i*3600000).toISOString()+","+(20+0.01*i+3*Math.sin(i*Math.PI/12)))
  await writeFile(data,rows.join("\n")+"\n")
  const raw={...overrides,schema_version:1,task:{dataset_path:data,time_col:"date",target_col:"target",frequency:"h",seq_len:24,pred_len:12,
    window_stride:12,split_mode:"ratio",train_ratio:0.6,val_ratio:0.2,test_ratio:0.2,model_families:["Naive","Linear"],reflection_enabled:false,...overrides.task},
    execution:{mode:"deterministic",max_experiments:4,...overrides.execution},
    decisions:{adjustment:{strategy:"off"},...overrides.decisions},memory:{enabled:false,...overrides.memory},
  }
  raw.task={...raw.task,...overrides.task};raw.execution={mode:"deterministic",max_experiments:4,...overrides.execution}
  const config=ConfigSchema.parse(raw)
  const configFile=path.join(dir,"task.json")
  await writeFile(configFile,JSON.stringify(config))
  return {dir,data,configFile,config,workspace:path.join(dir,"workspace"),cleanup:()=>rm(dir,{recursive:true,force:true})}
}

export function mockProvider(options:{evidence?:boolean;onGap?:(input:any)=>void;strategy?:"off"|"a_lightweight";invalidProposal?:boolean}={}) {
  const requests:any[]=[]
  let call=0
  const server=Bun.serve({hostname:"127.0.0.1",port:0,async fetch(req){
    const body=await req.json() as any;requests.push(body)
    const system=body.messages[0].content as string
    let content:any,tools:any[]|undefined
    const tool=(name:string,args:unknown={})=>{tools=[{id:"mock_"+(++call),type:"function",function:{name,arguments:JSON.stringify(args)}}]}
    if(!system.includes("FROZEN TASK AND CURRENT STATE:")) {
      const raw=body.messages[1].content as string
      const context=raw.includes("Blind window context:")?raw.split("Blind window context:").at(-1)!.trim():raw
      const jsonStart=context.indexOf("{")
      let input:any;try{input=JSON.parse(context.slice(jsonStart))}catch{throw new Error("Mock context was not JSON: "+raw.slice(0,300))}
      if(input.unused_evidence_candidate_summaries) {
        options.onGap?.(input)
        const gap=!!options.evidence && !input.state.evidences.length
        content={evidence_gap:gap?1:0,needed_evidence:gap?"Training pattern knowledge":"",evidence_coverage:gap?.5:1,missing_evidence:gap?["Scoped training knowledge"]:[],evidence_consistency:1,evidence_conflicts:[],evidence_confidence:.9,rationale:gap?"Need training knowledge before concluding":"The visible evidence supports preserving the committed prediction"}
        tool("StructuredOutput",content);content=undefined
      } else if(input.available_candidate_types) content={candidate_type:input.available_candidate_types[0].candidate_type,rationale:"Review the available scoped evidence"}
      else {
        const state=input.critic_prediction_state ?? input.critic_prediction_state_history?.initial ?? input
        const points=state.predictions??input.prediction_points??input.points??input.raw_prediction_window??input.predictions
        content={window_index:state.window_index??input.window_index,points:options.invalidProposal?[]:points?.map((p:any)=>({horizon_index:p.horizon_index,timestamp:p.timestamp,value:p.value??p.pre_reflection_baseline})),rationale:"Preserve the actual committed forecast after reviewing scoped evidence",evidence_ids:input.provisional_evidence_fusion?.fused_evidence_ids??[]}
      }
      if(content)content=JSON.stringify(content)
    } else {
      const matched=system.match(/FROZEN TASK AND CURRENT STATE:\n([\s\S]*?)\n\nFROZEN KNOWLEDGE:/)
      const state=JSON.parse(matched![1])
      const phase=state.state.phase
      if(phase==="pre-forecast") {
        if(!state.analysis) tool("analyze_dataset")
        else if(!body.messages.some((m:any)=>m.role==="assistant"&&m.tool_calls?.some((t:any)=>t.function.name==="pre_forecast_report")))
          tool("pre_forecast_report",{content:"## Task Summary\nForecast the configured target.\n## Data Characteristics\nTraining analysis demonstrates a periodic synthetic signal.\n## Domain Context\nNo unsupported business facts.\n## Risk Flags\nOnly training and permitted validation-front data inform decisions.\n## Modeling Directions\nCompare frozen allowed models using validation MSE."})
        else content="Pre-forecast analysis and actual report completed."
      } else if(phase==="skills-review") {
        if(!state.skills.length) tool("forecast_skill",{name:"periodic-baseline",description:"Forecast periodic hourly data using training-only patterns",content:"Compare the allowed models by held-out validation MSE. Keep all learned scaling within the training split and preserve the fitted model for testing."})
        else if(state.task.reflection_enabled && !body.messages.some((m:any)=>m.tool_calls?.some((t:any)=>t.function.name==="configure_window_policy"))) tool("configure_window_policy")
        else content="Task Skills validated and the configured window policy is frozen."
      } else if(phase==="forecast") {
        if(!state.experiments.length) tool("generate_model",{model:"Naive",params:{}})
        else if(!state.reflections[state.experiments[0].run_id]) tool("forecast_reflect",{run_id:state.experiments[0].run_id,decision:"keep",rationale:"A real baseline establishes validation performance before further decisions."})
        else if(!state.adjustment_decision && options.strategy) tool("decide_adjustment",{strategy:options.strategy,reason:"Use the explicitly chosen test strategy and bounded validation evidence."})
        else if(!state.search_done) tool("finish_search",{reason:"The completed baseline and required reflection establish an auditable selection."})
        else content="Search completed and validation selection is locked."
      } else {
        if(!state.test) tool("test_best")
        else if(state.task.reflection_enabled && !state.test_reflection) tool("run_window_reflection")
        else if(!state.final_report) tool("final_report",{content:"The real fitted baseline was selected on held-out validation and reused for test inference. Aggregate metrics describe this synthetic task. Window reasoning preserves or changes predictions only after deterministic checks. This mock-provider integration test verifies orchestration and does not measure a real LLM's reasoning quality."})
        else content="Actual test inference, required window processing and final report completed."
      }
    }
    return Response.json({choices:[{finish_reason:tools?"tool_calls":"stop",message:{role:"assistant",content:content??null,...(tools?{tool_calls:tools}:{})}}],usage:{total_tokens:100}})
  }})
  return {server,requests,baseURL:`http://127.0.0.1:${server.port}/v1`,close:()=>server.stop(true)}
}
