# -*- coding: utf-8 -*-
import pickle

from bottle import route, run, template
import webbrowser
from nltk.stem.wordnet import WordNetLemmatizer

from ner import StructuredNER
import nltk
from case.MosesTrueCaser import MosesTrueCaser

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
      $("#out").load("http://localhost:8086/ner/"+ encodeURI(t), function() {
        $("[rel=tooltip]").tooltip({html: true, placement: 'bottom'});
      });
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

    <script src="http://netdna.bootstrapcdn.com/twitter-bootstrap/2.1.1/js/bootstrap.min.js"></script>
    <link href="http://netdna.bootstrapcdn.com/twitter-bootstrap/2.1.1/css/bootstrap-combined.min.css" rel="stylesheet">


    <link rel="stylesheet" href="https://raw.github.com/drewwilson/TipTip/master/tipTip.css" type="text/css" charset="utf-8">

    <style type="text/css" media="screen">
    body {
        background: #fff;
        font-family:
    }
    #container {
        text-align: center;
    }
    #out {
        display: block;
    }
    #input, #out {
        outline: none;
        padding: 10px;
        display: block;
        margin-left: -200px;

        position: relative;
        left: 50%;
        width: 400px;
        height: 100px;

        background: white;
        border: 0;
        border-radius: 7px;
    }
    #input {
        border: 1px solid #f0f0f0;
    }
    #out {
        margin-top: 1em;

        line-height: 2em;

        position: relative;
        left: 50%;
        width: 700px;
        margin-left: -350px;

        height: auto;
    }
    .NE {
        border-radius: 3px;
        display: inline-block;
        background: #f0f0f0;
        margin-right: 3px;
        margin-bottom: 0.5em;
        padding: 2px;
    }

    .O {
        background: none;
    }

    .PER {

    }

    .MISC {

    }

    .LOC {

    }

    .ORG {

    }



    .ne_label {
        width: 100%;
        font-size: 80%;
        font-weight: bold;
        color: #666;
        text-align: center;
        clear: both;
        background: none;
        display: inline-block;
    }

    .features {
        display: none;
        position: absolute;
        bottom: 200px;
        left: 50%;
        margin-left: 100px;
        width: 100px;
    }

    .NE:hover .features {
        display: block;
    }
    </style>

</head>

<body>
<div id="container">
    <h1>Named Entity Recognition Demo.</h1>

    <textarea cols=​"60" id="input" rows=​"4">...​</textarea>​

    <div id="out"></div>
</div>

</body>
</html>


    """


tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+|[^\w\s]+')
tagger = nltk.data.load('taggers/maxent_treebank_pos_tagger/english.pickle')
truecaser = MosesTrueCaser(open('models/truecase/truecase-model.en'))
nerecognizer = StructuredNER(open("models/eng_gaz.pickle"), tokenizer, tagger, WordNetLemmatizer(), truecaser)

@route('/ner/:sent')
def ner(sent):
    s = ""

    for token, label, features in nerecognizer.recognize(sent):
        s += "<span class='NE %s'>" % label
        s += token

        f = ""
        for ft in features:
            f += ft.replace("<", "-").replace(">", "-") + "<br />"

        s += "<span class='ne_label' rel='tooltip' title='%s'>%s</span>" % (f, label)
        #s += "<div class='features'><strong>Features:</strong><br />"


        #s += "</div>"

        s += "</span>"

    return s


webbrowser.open('http://localhost:8086')
run(host='localhost', port=8086)
