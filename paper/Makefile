PROJECT=main

TEX=pdflatex -shell-escape
BUILDTEX=$(TEX) $(PROJECT).tex

all:
	$(BUILDTEX)

two:
	$(BUILDTEX) && \
	$(BUILDTEX)

clean:
	rm -f *.dvi *.log *.bak *.aux *.bbl *.idx *.ps *.eps *.out *.cut *.blg *.log *~

