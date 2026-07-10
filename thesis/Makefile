# makefile pro preklad LaTeX verze Bc. prace
# makefile for compilation of the thesis
# (c) 2008 Michal Bidlo
# E-mail: bidlom AT fit vut cz
# Edited by: dytrych AT fit vut cz
#===========================================
# asi budete chtit prejmenovat / you will probably rename:
CO=speech_music_classification
BUILD=build

# Usage: make pdf PRINT=1   or   make print
PRINT ?= 0
ifeq ($(PRINT),1)
TEX_ENTRY := $(CO)_print.tex
# Keep .aux/.bbl names as $(CO) so bibtex matches
PDFLATEX_JOB := -jobname=$(CO)
else
TEX_ENTRY := $(CO).tex
PDFLATEX_JOB :=
endif

.PHONY: all pdf print live clean

all: pdf

pdf:
	mkdir -p $(BUILD)
	pdflatex -output-directory=$(BUILD) $(PDFLATEX_JOB) $(TEX_ENTRY)
	-bibtex $(BUILD)/$(CO)
	# makeglossaries -d $(BUILD) $(CO)
	# makeglossaries-lite -d $(BUILD) $(CO)
	pdflatex -output-directory=$(BUILD) $(PDFLATEX_JOB) $(TEX_ENTRY)
	pdflatex -output-directory=$(BUILD) $(PDFLATEX_JOB) $(TEX_ENTRY)
	cp $(BUILD)/$(CO).pdf $(CO).pdf

print:
	$(MAKE) pdf PRINT=1

live:
	mkdir -p $(BUILD)
# -g forces an initial rebuild so switching between PRINT=1 and PRINT=0
# always reflects the new option (otherwise latexmk sees the cached PDF
# from the previous mode and skips compilation).
ifeq ($(PRINT),1)
	latexmk -g -pdf -pvc -output-directory=$(BUILD) -jobname=$(CO) $(CO)_print.tex
else
	latexmk -g -pdf -pvc -output-directory=$(BUILD) $(CO).tex
endif

clean:
	rm -rf $(BUILD)
	rm -f *~
	rm -f $(CO).pdf
	rm -f $(CO).aux $(CO).bbl $(CO).blg $(CO).log $(CO).out $(CO).toc \
	      $(CO).lof $(CO).lot $(CO).ptc $(CO).fls $(CO).fdb_latexmk \
	      $(CO).synctex.gz $(CO).bcf $(CO).run.xml

pack:
	tar czvf $(CO).tar.gz *.tex *.bib *.bst ./template-fig/* ./bib-styles/* ./cls/* zadani.pdf $(CO).pdf Makefile Changelog

rename:
	mv $(CO).tex $(NAME).tex
	mv $(CO)-01-kapitoly-chapters.tex $(NAME)-01-kapitoly-chapters.tex
	mv $(CO)-01-kapitoly-chapters-en.tex $(NAME)-01-kapitoly-chapters-en.tex
	mv $(CO)-20-literatura-bibliography.bib $(NAME)-20-literatura-bibliography.bib
	mv $(CO)-30-prilohy-appendices.tex $(NAME)-30-prilohy-appendices.tex
	mv $(CO)-30-prilohy-appendices-en.tex $(NAME)-30-prilohy-appendices-en.tex
	sed -i "s/$(CO)-01-kapitoly-chapters/$(NAME)-01-kapitoly-chapters/g" $(NAME).tex
	sed -i "s/$(CO)-01-kapitoly-chapters-en/$(NAME)-01-kapitoly-chapters-en/g" $(NAME).tex
	sed -i "s/$(CO)-20-literatura-bibliography/$(NAME)-20-literatura-bibliography/g" $(NAME).tex
	sed -i "s/$(CO)-30-prilohy-appendices/$(NAME)-30-prilohy-appendices/g" $(NAME).tex
	sed -i "s/$(CO)-30-prilohy-appendices-en/$(NAME)-30-prilohy-appendices-en/g" $(NAME).tex
	sed -i "s/$(CO)/$(NAME)/g" Makefile

# Pozor, vlna neresi vse (viz popis.txt) / Warning - vlna is not solving all problems (see description.txt)
vlna:
	vlna -l $(CO)-*.tex

# Spocita normostrany / Count of standard pages
normostrany:
	echo "scale=2; `detex -n $(CO)-[01]*.tex | sed s/"^ *"/""/ | sed s/"^	*"/""/ | wc -c`/1800;" | bc
