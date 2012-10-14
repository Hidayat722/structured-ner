# -*- coding: utf-8 -*-
import pickle

from bottle import route, run, template
import webbrowser

from ner import StructuredNER
import nltk

@route('/')
def index():
    return """

<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
	"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">

<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
	<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>

	<title>NER Demo</title>

	<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.8.2/jquery.min.js"></script>
	<script type="text/javascript" charset="utf-8">
	function update() {
	  var t = $("textarea#input").val();
	  $("#out").load("http://localhost:8086/ner/"+ encodeURI(t));
	}

    $(document).ready(function() {
	//setup before functions
	var typingTimer;                //timer identifier
	var doneTypingInterval = 700;  //time in ms, 5 second for example

	//on keyup, start the countdown
	$('#input').keyup(function(){
	    typingTimer = setTimeout(update, doneTypingInterval);
	});

	//on keydown, clear the countdown
	$('#input').keydown(function(){
	    clearTimeout(typingTimer);
	});
	});

	</script>

</head>

<body>
<textarea cols=​"60" id="input" rows=​"4">​</textarea>​

<div id="out">a</div>

</body>
</html>

    """


tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+|[^\w\s]+')
tagger = pickle.load(open("../taggers/alpino_ubt.pickle"))
nerecognizer = StructuredNER(open("ner_model.pickle"), tokenizer, tagger)

print

@route('/ner/:sent')
def ner(sent):
    return " ".join(map("/".join, nerecognizer.recognize(sent)))


webbrowser.open('http://localhost:8086')
run(host='localhost', port=8086)
