# SDSC_RAG
This is a proof of concept for how we might be able to combine three main components:

1) geospatial data: this will ideally be a publically accessible satellite remote sensing product with a suffeciently long time series. The example here mentions urban heat, and since temperature is one of the more robust and well documented data products, I will probably start there. Though preipitation and snow-cover are also good ideas for switzerland. 

2) a knowledge graph (KG) that is capable of interlinking different data entities with their semantic relationships to a large language model (LLM). I think we can lean on the (usually) well documented metadata fot these large geospatial layers, especially those that are derived from the same same sensors/satellites. 

3) a RAG (response augmented generation) module that can ground our LLM in external data, allowing it to retrieve specific, up-to-date data instead of relying solely on static training data, providing context-aware responses. 



A bit about my approach:

I probably should have chosen a prompt where I could best display my existing skills since my experience with KG's, RAG, and LLM's and is limited to a course I took on boot.dev. However, I have somewhat more expereince retrieving, combining and analyzing geospatial satellite remote sensing data for global inference, so the utility of using LLM's for a dashboard is clear to me.


