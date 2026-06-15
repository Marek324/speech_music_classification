# usage: make            (builds prezentace.pdf)
#        make live       (latexmk -pvc watch loop)
#        make clean      (removes aux files)

CO=prezentace
BUILD=build

.PHONY: all pdf live clean

all: pdf

pdf:
	mkdir -p $(BUILD)
	latexmk -pdf -output-directory=$(BUILD) $(CO).tex
	cp $(BUILD)/$(CO).pdf $(CO).pdf

live:
	mkdir -p $(BUILD)
	latexmk -g -pdf -pvc -output-directory=$(BUILD) $(CO).tex

clean:
	rm -rf $(BUILD)
	rm -f $(CO).pdf
