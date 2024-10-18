#!/bin/bash
pip install -r requirements.txt
wget https://storage.googleapis.com/pneuma_open/chroma.tar
tar -xf chroma.tar
rm chroma.tar
cd ./chroma
git init
cd ..
pip install ./chroma/