## BLAST it!

In order to tackle bio-terrorism, we've created a novel pipeline:
1. Produce output from GENERator or Evo 2.
2. Check it against the industry-standard BLAST database of known pathogens.
3. Determine likelihood of it being a pathogen using the PathoLM (Pathogen Language Model).
4. Use linear or non-linear probing for a final check.

### Probing Dashboard

View the interactive [Evo2 Probing Dashboard](https://htmlpreview.github.io/?https://github.com/marapowney/Varsity26BioGaurdrails/blob/main/evo2_probe/probe/dashboard/index.html) to explore probe performance across all 32 Evo2 7B layers.