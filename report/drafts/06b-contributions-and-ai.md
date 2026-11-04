# 7. Contributions

This was a solo project. All of the work is mine — design, implementation, experiments, analysis and this report — carried out with the AI assistance disclosed in §8. The report says "we" by convention.

# 8. Use of AI tools

## 8.1 How I used AI

I used Claude (Anthropic) throughout. It worked in a chat linked to my project folder, so it could read and write the repository and run smoke tests in a Linux shell on my PC. It wrote most of the code, the analysis scripts and the first drafts of this report.

I directed the project and made its decisions:
- **Working rules.** I gave the AI four working rules drawn from the feedback on my PA1 submission: test cache explanations rather than asserting them; treat tuning parameters as experimental variables; keep the harness design; fix loose ends from the start. Those rules are why the tile and cutoff are measured rather than chosen, why the drift monitor exists, and why the surprises in §5 were tested, not just explained.
- **Workflow and platform.** I set the workflow so that every change came to me as a staged edit, which I reviewed, committed and pushed myself. I chose native Windows/MSVC as the benchmark platform, and I required each experiment to report an estimated and an actual runtime.
- **Predictions.** The predictions in §2 were developed with the AI: it drafted the cost-model reasoning, and I reviewed each one, adopted all fourteen as my own commitments, and committed them before any measurement.
- **Running the experiments.** I ran every build, test and benchmark, and controlled the machine's conditions for the official runs.
- **Follow-ups.** When results disagreed with predictions, I decided to test the explanation before writing it up (E9), and then to add E9b rather than leave an unexplained artefact in the data.
- **The report.** I read the full draft and directed the final edits. The abstract is my own wording.

## 8.2 Where the AI was wrong, and what I did about it

1. **A convincing analysis that was wrong.** Three of the predictions I adopted were wrong (P6, P8, P10). The tile-size argument in particular read like a textbook. Committing the predictions to git before measuring is what exposed this, and §5 tests the corrected explanations. The lesson for me: agreeing with a well-argued prediction is not the same as having reasoned it out independently.
2. **A flaw in the benchmark design.** The AI's first harness ran E6's pressure levels one after another. When additional workload ran on my PC during the first full run, 16–20% drift appeared, and that ordering could not separate it from cache effects. We added a drift monitor and round-robin ordering. Later, the replacement rotation scheme produced a 2–5% carry-over bias that nobody had anticipated, and E9b isolated it.
3. **Confident fixes that didn't work.** The AI's build instructions failed at the link step: the Visual Studio terminal was a 32-bit environment. Its first fix didn't take effect either. I stopped further edits and had the cause confirmed first, which four read-only commands did. Separately, one automated edit corrupted a source file into about 700,000 lines. The build caught it, the file was restored from git, and every later edit used search text that had to match exactly once.
4. **Wrong numbers and a wrong citation.** Checking the draft against the CSVs found five wrong figures. For example, the draft said Strassen exceeded the classical error bound in every configuration; it was 10 of 11. It also cited a Higham theorem number that does not exist: the bound is Brent's, proved in Higham §23.2.2. Every number in §4–5 was then re-derived from the data.

## 8.3 AI as the "second team member" on a solo project

With no teammates, the AI filled the roles a group would share.

**Where it helped.** It generated cross-platform plumbing that would have taken me days (CMake presets, a portable random number generator, Windows environment capture, plotting). It explained mechanisms well enough for me to judge them.

**Where it threatened to hurt.** The risk was speed. It produced code and prose faster than I could fully absorb, which makes it tempting to accept a convincing analysis without understanding it. The project's safeguards were checks that don't depend on how persuasive the text is: tests and a correctness gate, predictions committed before measurement, the drift monitor, scripts that regenerate every number, and a page-by-page check of the final document. On top of those, I held to one working rule throughout: when something looked wrong, stop and confirm the cause before changing anything else. Between them, these caught every error listed in §8.2.
