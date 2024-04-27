wget --header "X-Ms-Version: 2019-12-12" https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar

tar -xvf msmarco_v2_passage.tar

for file in msmarco_v2_passage/*.gz; do
    gzip -d "$file"
done

