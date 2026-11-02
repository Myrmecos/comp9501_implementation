import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const deck = await PresentationFile.importPptx(await FileBlob.load('output/hfa_shfa_presentation.pptx'));
await fs.mkdir('tmp/final-slides',{recursive:true});
async function save(path, blob){await fs.writeFile(path,new Uint8Array(await blob.arrayBuffer()));}
for (const [i, slide] of deck.slides.items.entries()) await save(`tmp/final-slides/slide-${String(i+1).padStart(2,'0')}.png`,await deck.export({slide,format:'png',scale:1}));
await save('tmp/final-slides/montage.webp',await deck.export({format:'webp',montage:true}));
console.log(deck.slides.items.length);
