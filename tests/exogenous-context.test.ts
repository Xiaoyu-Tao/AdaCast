import {test,expect} from "bun:test"
import {buildUnifiedEvidenceReasoningInput,needsInitialEvidenceAssessmentRetry} from "../src/domain/forecast-window-reflection"
import {withDomain,Session,SessionPrompt} from "../src/domain/runtime"
import {AppError} from "../src/contracts"
import {checkAbort} from "../src/util"

test("an unassessed cancelled prior must be assessed before evidence seeking resumes",()=>{
  const state:any={version:1,window_index:0,evidences:[],predictions:[],evidence_gap:1,needed_evidence:"The initial evidence-gap assessment failed; additional relevant evidence may still be needed."}
  expect(needsInitialEvidenceAssessmentRetry(state)).toBe(true)
  expect(needsInitialEvidenceAssessmentRetry({...state,needed_evidence:"Confirm the holiday regime"})).toBe(false)
  expect(needsInitialEvidenceAssessmentRetry({...state,evidences:[{}]})).toBe(false)
  expect(needsInitialEvidenceAssessmentRetry({...state,evidence_gap:0})).toBe(false)
})

test("unified context fusion honors an explicit recent exogenous suffix while retaining all load history",()=>{
  const stamps=Array.from({length:864},(_,index)=>new Date(Date.UTC(2025,8,23)+index*900000).toISOString())
  const target={columns:["timestamp","observed_value"],rows:stamps.slice(0,711).map(timestamp=>[timestamp,1]),row_count:711}
  const table=(timestamps:string[])=>({columns:["timestamp","city_temperature"],rows:timestamps.map(timestamp=>[timestamp,20]),row_count:timestamps.length})
  const candidate:any={candidate_type:"context",evidence_candidate_id:"context:current-window",payload:{
    lookback_window_exogenous:table(stamps.slice(711-288,711)),predict_window_exogenous:table(stamps.slice(768,864))}}
  const state:any={version:1,window_index:0,evidences:[],predictions:stamps.slice(768,864).map((timestamp,horizon_index)=>({horizon_index,timestamp,value:1})),evidence_gap:1,needed_evidence:"context"}
  const input={state,initialState:state,selectedCandidate:candidate,allCandidates:{context:candidate},lookbackTargetWindow:target,experienceRuleReference:null}
  const fused=buildUnifiedEvidenceReasoningInput({...input,exogenousHistorySteps:288})
  expect(fused.lookback_window_target).toBe(target)
  expect((fused.provisional_evidence_fusion.evidence_attachments[0].auxiliary.lookback_window_exogenous as any).row_count).toBe(288)
  expect(()=>buildUnifiedEvidenceReasoningInput(input)).toThrow("incomplete")
  expect(()=>buildUnifiedEvidenceReasoningInput({...input,exogenousHistorySteps:712})).toThrow("scope is invalid")
  candidate.payload.lookback_window_exogenous=table(stamps.slice(711-287,711))
  expect(()=>buildUnifiedEvidenceReasoningInput({...input,exogenousHistorySteps:288})).toThrow("incomplete")
  candidate.payload.lookback_window_exogenous=table(stamps.slice(710-288,710))
  expect(()=>buildUnifiedEvidenceReasoningInput({...input,exogenousHistorySteps:288})).toThrow("not aligned")
  candidate.payload.lookback_window_exogenous=table(stamps.slice(0,711))
  expect(()=>buildUnifiedEvidenceReasoningInput(input)).not.toThrow()
})

test("cancelling one isolated child does not raise a fatal parent cancellation",async()=>{
  let fatal=0
  const signal=new AbortController().signal
  const scope:any={config:{},store:{set(){},append(){}},signal,onFatal(){fatal++},llm:{complete(_m:any,_t:any,_p:any,childSignal:AbortSignal){
    return new Promise((_resolve,reject)=>childSignal.addEventListener("abort",()=>{try{checkAbort(childSignal)}catch(error){reject(error)}},{once:true}))
  }}}
  await withDomain(scope,async()=>{
    const child=await Session.create({parentID:"test",title:"test",permission:{}})
    const pending=SessionPrompt.prompt({sessionID:child.id,messageID:"test",model:{providerID:"test",modelID:"test"},agent:"test",system:"test",tools:{},parts:[]})
    await SessionPrompt.cancel(child.id)
    await expect(pending).rejects.toThrow("Execution interrupted")
  })
  expect(fatal).toBe(0)
  expect(signal.aborted).toBe(false)
  scope.llm.complete=async()=>{throw new AppError("BUDGET_EXHAUSTED","budget",5)}
  await withDomain(scope,async()=>{
    const child=await Session.create({parentID:"test",title:"budget",permission:{}})
    await expect(SessionPrompt.prompt({sessionID:child.id,messageID:"test",model:{providerID:"test",modelID:"test"},agent:"test",system:"test",tools:{},parts:[]})).rejects.toThrow("budget")
  })
  expect(fatal).toBe(1)
})
