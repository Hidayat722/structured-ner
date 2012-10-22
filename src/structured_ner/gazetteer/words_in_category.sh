#! /bin/sh

curl -H "Accept: text/xml" "http://en.wiktionary.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:$1&cmlimit=500&format=xml" | egrep -o '<cm pageid="[0-9]+" ns="[0-9]+" title="[^"]+"' | sed 's/.* title=\"\([^"]*\)\"$/\1/g' | grep -v ^Category: > $1
