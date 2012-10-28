SHELL = /bin/bash

models/truecase_:
	#mkdir -p models/truecase
	pushd /tmp; \
	wget http://www.statmt.org/europarl/v7/nl-en.tgz; \ 
	wget http://www.statmt.org/europarl/v7/es-en.tgz; \
	wget http://www.statmt.org/europarl/v7/de-en.tgz; \
	tar -zxvf nl-en.tgz; tar -zxvf es-en.tgz; tar -zxvf de-en.tgz; \
	popd
	
	cd case/moses; \
	perl tokenizer.perl nl < /tmp/europarl-v7.nl-en.nl > /tmp/nl.tok; \
	perl tokenizer.perl en < /tmp/europarl-v7.nl-en.en > /tmp/en.tok; \
	perl tokenizer.perl es < /tmp/europarl-v7.nl-en.es > /tmp/es.tok; \
	perl tokenizer.perl de < /tmp/europarl-v7.nl-en.de > /tmp/de.tok; \
	perl train-truecaser.perl --model ../../models/truecase/en --corpus /tmp/en.tok; \
	perl train-truecaser.perl --model ../../models/truecase/nl --corpus /tmp/nl.tok; \
	perl train-truecaser.perl --model ../../models/truecase/es --corpus /tmp/es.tok; \
	perl train-truecaser.perl --model ../../models/truecase/de --corpus /tmp/de.tok; \
