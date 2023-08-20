let pages = 1;
let recordsAll = [];
var working = false;
async function start_agritrop_import(resumptionToken) {
    if (!resumptionToken) {
        recordsAll = [];
    }
    axios.get("https://agritrop.cirad.fr/cgi/oai2?verb=ListRecords&" + (
        resumptionToken ? "resumptionToken=" + resumptionToken :
            "metadataPrefix=oai_dc&set=CTS_2_2019"))
        .then((result) => {
            var result = convert.xml2json(result.data, { compact: true, spaces: 4 });
            var records = JSON.parse(result)["OAI-PMH"]["ListRecords"].record;
            var resumptionToken = JSON.parse(result)["OAI-PMH"]["ListRecords"].resumptionToken;
            //datas.items = [];
            //JSON.parse(result1)["OAI-PMH"]["ListRecords"].channel.item;
            //var objectResults = JSON.parse(result.data);
            recordsAll = recordsAll.concat(records);
            if (resumptionToken) {
                pages++;
                console.log("resum:" + pages, resumptionToken);
                start_agritrop_import(resumptionToken._text);
            }
            else {
                console.log("recordsAll", recordsAll.length);
                do_agritrop_import(recordsAll)
                console.log("no resumptionToken")
            }
        })
        .catch((err) => {
            console.log("err bowl api", err);
        })
}