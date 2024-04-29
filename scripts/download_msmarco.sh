wget --header "X-Ms-Version: 2019-12-12" https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar -P ../data

tar -xvf ../data/msmarco_v2_passage.tar -C ../data/

for file in ../data/msmarco_v2_passage/*.gz; do
    gzip -d "$file"
done

