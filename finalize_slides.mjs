import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const skill='/Users/entomophile/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const { finalizePresentation } = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
await fs.mkdir('.codex-finalizer',{recursive:true}); await fs.mkdir('output',{recursive:true});
const result=await finalizePresentation({
 workspaceDir:process.cwd(), candidatePath:path.join(process.cwd(),'tmp/hfa_shfa_presentation_candidate.pptx'), finalPath:path.join(process.cwd(),'output/hfa_shfa_presentation.pptx'),
 pythonExecutable:'/Users/entomophile/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:12, requiredNativeTableOwnerSlides:[],
 fontPolicy:{basis:'design',families:['Aptos']}, verifyArtifactToolImport:true,
 receiptPath:path.join(process.cwd(),'.codex-finalizer/hfa_shfa_presentation.validation.json')
}); console.log(JSON.stringify(result,null,2));
