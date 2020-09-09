rm -rf bin/*.py*
python3 -m compileall -b src
mv src/*.pyc bin/
cp src/demo.py bin/
