# -*- coding: utf-8 -*-

from bottle import route, run, template
import webbrowser
from nltk import word_tokenize, pos_tag

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
	</script>

</head>

<body>
<textarea cols=​"60" id="input" onkeyup="javascript:update();" rows=​"4">​</textarea>​

<div id="out">a</div>

</body>
</html>

    """

@route('/ner/:sent')
def ner(sent):
    return " ".join(map("/".join, pos_tag(word_tokenize(sent))))


webbrowser.open('http://localhost:8086')
run(host='localhost', port=8086)
