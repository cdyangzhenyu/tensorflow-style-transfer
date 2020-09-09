mkdir style_transfer
cp -r bin style_transfer/
cp -r banner style_transfer/
cp -r models style_transfer/
cp startup.sh style_transfer/
cp install.sh style_transfer/
cp INSTALL.md style_transfer/
tar -zcvf style_transfer.tar.gz style_transfer
rm -rf style_transfer
