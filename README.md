<h3>Model Deployment Strategies</h3>
<br>While the adoption of machine learning and deep learning techniques continue to grow, many organizations find it difficult to actually deploy these sophisticated models into production. It is common to see data scientists build powerful models, yet these models are not deployed because of the complexity of the technology used or lack of understanding related to the process of pushing these models into production.
<br>
<br>As part of this talk, I will review several deployment design patterns for both real-time and batch use cases. I’ll show how these models can be deployed as scalable, distributed deployments within the cloud, scaled across hadoop clusters, as APIs, and deployed within streaming analytics pipelines. I will also touch on topics related to security, end-to-end governance, pitfalls, challenges, and useful tools across a variety of platforms. This presentation will involve demos and sample code for the the deployment design patterns.
<br>
<br>This repo contains a few of my ideas, strategies, and code for deploying your models in production and at scale.
<br>
<br><h3>So where can machine learning models be deployed?</h3>
<br>&nbsp;&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;Deploy as a Batch Process
<br>&nbsp;&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;Deploy as a Web Service (REST API)
<br>&nbsp;&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;Deploy within Web App
<br>&nbsp;&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;Deploy online as part of a real-time data stream
<br>&nbsp;&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;Deploy/Embed within Devices
<br>
<br><h3>Based on these deployments, here are high-level flows for several deployment designs (including open source tech being used): </h3>
NOTE: A deeper-dive design ref architecture is in progress for each
<br>
<br><b>ML Deployment Design - Batch</b>
<br><img src="images/ml_deployment_batch.png" class="inline"/>
<br>
<br><b>ML Deployment Design - Batch</b>
<br><img src="images/ml_deployment_batch_dsx_hdp_livy.png" class="inline"/>
<br>
<br><b>ML Deployment Design - Batch</b>
<br><img src="images/ml_deployment_batch_dsx_hdp.png" class="inline"/>
<br>
<br><b>ML Deployment Design - API Endpoint</b>
<br><img src="images/ml_deployment_dsx_api.png" class="inline"/>
<br>
<br><b>ML Deployment Design - Real-time, Streaming Analytics</b>
<br><img src="images/ml_deployment_streaming_sam.png" class="inline"/>
<br>
<br><b>References:</b>
<br>&nbsp;&nbsp;&nbsp;&nbsp;&bull;&nbsp;<a href="https://allthingsopen.org/talk/deployment-design-patterns-deploying-machine-learning-and-deep-learning-models-into-production/">Presented these concepts at the AllThingsOpen Conference</a>
