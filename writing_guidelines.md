# Guideline to Technical Writing
note: this guide gives credit to the staff of MIT 2.671, specifically Dr. Barbara Hughey, for many of the points presented here.

Learning to communicate clearly is a very important part of all science and engineering, both in industry and research. Your reports in MEAM 5200, while not necessarily presenting new or groundbreaking information, will nonetheless be graded on writing clarity, style, and quality for both your benefit (as good practice) and ours (as graders).

### General guidelines
- Tonally, you should be aiming for formal without uses of "I" or other informal markers. At the same time, beware of abusing passive voice: your reader should always come away knowing who/what did what.
- Do your due diligence in making figures and making sure that data is presented and graphed correctly, including axis labels. Even after following lab instructions, all data/results should pass a "common sense" check (ex. measuring in distances traveled in inches but using 9.81 as the gravitational constant should set off alarms in your head).
- Try to state things as succinctly as possible. If cutting words/phrases leaves the meaning intact, cut them.
- If this is your first foray into LaTeX, the Overleaf documentation is helpful: [https://www.overleaf.com/learn](https://www.overleaf.com/learn).
- Technical writing is a key skill no matter where you go from this class, and you can't learn it by generating it with an LLM! Take pride in your work.

## Graphics and Visual Communication
Any figures or graphics included in your report should be introduced (mentioned) in the text of the paper, before the reader gets to the graphic but nearby. In LaTeX, you can make sure figures appear near the place you put them in the .tex by using:
- `\begin{figure}[h]` ("here") -- this is a suggestion to the LaTeX renderer, and it will try to keep your figure on the current page unless it violates layout rules / doesn't fit
- `\begin{figure}[t]` ("top of page") -- put the figure roughly on the top of a page
- `\begin{figure}[b]` ("bottom of page") -- put the figure roughly on the bottom of a page
- `\begin{figure}[!h]` ("override, here") -- override layout rules to place the figure here
- You can combine these options (ex. `[!ht]` "override, here or top of page").

It is common for people to scan a paper and look at the figures before reading the paper itself in depth. Because of this, **we want to make figures capable of standing alone**, that is, the audience can understand the message and point of the figure just by looking at it and reading the caption. This leads to a few guidelines:
1. Make sure that all plots or graphs have axes and titles clearly labeled, in an easy-to-read font and font size, with any necessary legends (making sure they don't obscure the data).
2. Photos / images should be well-exposed and easy to understand: but default to diagrams if describing a setup or concept, as they are easier to read.
3. Diagrams should be labeled clearly with arrows if necessary.
4. Captions should say more than "Experiment" -- they should describe the figure, and call attention to the main takeaway.

Some formatting rules of thumb:
- Figure captions are placed immediately _below_ the figure they refer to.
- Table captions are placed immediately _above_ the table.
- Font of caption should be smaller than the body text, to visually distinguish.

## Method / Experimental Design

This section tells the reader **what you did**, describes your **experiment setup** and your **data acquisition and analysis methods.** It should describe your experiments in sufficient detail to convince readers your methods are sound, so that results can be understood, and so that the reader could replicate your work: a cornerstone of scientific research is reproducibility by your peers.

- This section is not like a lab manual that explains step by step your entire process. It should assume the reader is familiar with methods used in this field, but be enough details for the reader to be able to repeat your experiments. Most important information and details should come first, followed by other details that are important to each step of your results and discussion -- this is most likely **not** the order in which you performed the steps.
- Use subheadings or subsections with meaningful and relevant titles to help the reader find information more easily.
- List all numbers with **units** and uncertainty if specified or computed.
- Provide clear diagrams of the setup if relevant.

## Evaluation and Analysis / Results and Discussion

- Remind readers of your experimental goals as stated in the introduction, and present all relevant data and meaning of results.
- Analyze experimental problems and shortcomings, as well as unexpected results. Discuss limitations in analysis.
- Data and results should always have accompanying commentary, and should not just be thrown in without further explanation.
- Refer to the Graphics section for more guidelines on figures specifically

Results are written in the past tense; discussion points are often addressed in the present tense---a simple rule is to use past tense whenever describing past experimental action and present tense for what's always true.

Stylistic things to avoid:
- "Figure 2 clearly shows..." and "Obviously..." -- if it's clear, then just state it. If it needs an explanation, explain it.
- Hedging excessively, e.g. "The causes of discrepancies between literature and results is _unknown_, but _could possibly_ be caused by a _presumable_ error in calibration, _either_ human or wear-induced."
- Stating "experimenter error" to explain a discrepancy - be specific if possible about what you think went wrong.

**In this class, we will often provide specific questions to answer in your report -- many of those questions should be addressed in the discussion portions of these sections, and we will be looking for them when grading.**