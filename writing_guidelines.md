# Guideline to Technical Writing

Learning to communicate clearly is a very important part of all science and engineering, both in industry and research. Your reports in MEAM 520, while not necessarily presenting new or groundbreaking information, will nonetheless be graded on writing clarity, style, and quality for both your benefit (as good practice) and ours (as graders).

## Graphics for Papers
Any figures or graphics included in your report should be introduced (mentioned) in the text of the paper, before the reader gets to the graphic but nearby. In LaTeX, you can make sure figures appear near the place you put them in the .tex by using:
- `\begin{figure}[h]` ("here") -- this is a suggestion to the LaTeX renderer, and it will try to keep your figure on the current page unless it violates layout rules / doesn't fit
- `\begin{figure}[t]` ("top of page") -- put the figure roughly on the top of a page
- `\begin{figure}[b]` ("bottom of page") -- put the figure roughly on the bottom of a page
- `\begin{figure}[!h]` ("override, here") -- override layout rules to place the figure here
- You can combine these options (ex. `[!ht]` "override, here or top of page").

It is common for people to scan a paper and look at the figures before reading the paper itself in depth. Because of this, **we want to make figures standalone**, that is, the audience can understand the message and point of the figure just by looking at it and reading the caption. This leads to a few guidelines:
1. Make sure that all plots or graphs have axes and titles clearly labeled, in an easy-to-read font and font size, with any necessary legends (making sure they don't obscure the data).
2. Photos / images should be well-exposed and easy to understand: but default to diagrams if describing a setup or concept, as they are easier to read.
3. Diagrams should be labeled clearly with arrows if necessary.
4. Captions should say more than "Experiment" -- they should describe the figure, and call attention to the main takeaway.

Some formatting rules of thumb:
- Figure captions are placed immediately _below_ the figure they refer to.
- Table captions are placed immediately _above_ the table.
- Font of caption should smaller than the body text, to visually distinguish.

