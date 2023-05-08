"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[837],{3905:(e,t,a)=>{a.d(t,{Zo:()=>d,kt:()=>s});var n=a(7294);function r(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function i(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function l(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?i(Object(a),!0).forEach((function(t){r(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):i(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function o(e,t){if(null==e)return{};var a,n,r=function(e,t){if(null==e)return{};var a,n,r={},i=Object.keys(e);for(n=0;n<i.length;n++)a=i[n],t.indexOf(a)>=0||(r[a]=e[a]);return r}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(n=0;n<i.length;n++)a=i[n],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(r[a]=e[a])}return r}var m=n.createContext({}),p=function(e){var t=n.useContext(m),a=t;return e&&(a="function"==typeof e?e(t):l(l({},t),e)),a},d=function(e){var t=p(e.components);return n.createElement(m.Provider,{value:t},e.children)},u="mdxType",g={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},c=n.forwardRef((function(e,t){var a=e.components,r=e.mdxType,i=e.originalType,m=e.parentName,d=o(e,["components","mdxType","originalType","parentName"]),u=p(a),c=r,s=u["".concat(m,".").concat(c)]||u[c]||g[c]||i;return a?n.createElement(s,l(l({ref:t},d),{},{components:a})):n.createElement(s,l({ref:t},d))}));function s(e,t){var a=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var i=a.length,l=new Array(i);l[0]=c;var o={};for(var m in t)hasOwnProperty.call(t,m)&&(o[m]=t[m]);o.originalType=e,o[u]="string"==typeof e?e:r,l[1]=o;for(var p=2;p<i;p++)l[p]=a[p];return n.createElement.apply(null,l)}return n.createElement.apply(null,a)}c.displayName="MDXCreateElement"},9540:(e,t,a)=>{a.r(t),a.d(t,{assets:()=>m,contentTitle:()=>l,default:()=>g,frontMatter:()=>i,metadata:()=>o,toc:()=>p});var n=a(7462),r=(a(7294),a(3905));const i={},l="Optimizer",o={unversionedId:"modules/Optimizer",id:"modules/Optimizer",title:"Optimizer",description:"When using an optimizer, you need to configure the name item to indicate which optimizer to use, and then configure their respective parameters according to different optimizer. The following are the names of each optimizer.",source:"@site/docs/modules/Optimizer.md",sourceDirName:"modules",slug:"/modules/Optimizer",permalink:"/damo-embedding/docs/modules/Optimizer",draft:!1,editUrl:"https://github.com/uopensail/damo-embedding/edit/docs/docs/docs/modules/Optimizer.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Initializer",permalink:"/damo-embedding/docs/modules/Initializer"},next:{title:"Learning Rate Scheduler",permalink:"/damo-embedding/docs/modules/Scheduler"}},m={},p=[{value:"SGD",id:"sgd",level:2},{value:"FTRL",id:"ftrl",level:2},{value:"Adagrad",id:"adagrad",level:2},{value:"Adam",id:"adam",level:2},{value:"AdamW",id:"adamw",level:2},{value:"Lion",id:"lion",level:2},{value:"Example",id:"example",level:2}],d={toc:p},u="wrapper";function g(e){let{components:t,...a}=e;return(0,r.kt)(u,(0,n.Z)({},d,a,{components:t,mdxType:"MDXLayout"}),(0,r.kt)("h1",{id:"optimizer"},"Optimizer"),(0,r.kt)("p",null,"When using an optimizer, you need to configure the ",(0,r.kt)("inlineCode",{parentName:"p"},"name")," item to indicate which optimizer to use, and then configure their respective parameters according to different optimizer. The following are the names of each optimizer."),(0,r.kt)("table",null,(0,r.kt)("thead",{parentName:"table"},(0,r.kt)("tr",{parentName:"thead"},(0,r.kt)("th",{parentName:"tr",align:null},"Optimizer"),(0,r.kt)("th",{parentName:"tr",align:null},"name"))),(0,r.kt)("tbody",{parentName:"table"},(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:null},"SGD"),(0,r.kt)("td",{parentName:"tr",align:null},"sgd")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:null},"FTRL"),(0,r.kt)("td",{parentName:"tr",align:null},"ftrl")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:null},"Adagrad"),(0,r.kt)("td",{parentName:"tr",align:null},"adagrad")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:null},"Adam"),(0,r.kt)("td",{parentName:"tr",align:null},"adam")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:null},"AdamW"),(0,r.kt)("td",{parentName:"tr",align:null},"adamw")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:null},"Lion"),(0,r.kt)("td",{parentName:"tr",align:null},"lion")))),(0,r.kt)("h2",{id:"sgd"},"SGD"),(0,r.kt)("p",null,(0,r.kt)("a",{parentName:"p",href:"https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD"},"SGD")," configure the following parameters:"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},"$\\gamma$: learning rate, default: 1e-3, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"gamma")),(0,r.kt)("li",{parentName:"ol"},"$\\lambda$: weight decay, default: 0, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"lambda"))),(0,r.kt)("h2",{id:"ftrl"},"FTRL"),(0,r.kt)("p",null,(0,r.kt)("a",{parentName:"p",href:"https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/37013.pdf"},"FTRL")," configure the following parameters:"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},"$\\alpha$: learning rate, default: 5e-3, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"gamma")),(0,r.kt)("li",{parentName:"ol"},"$\\beta:$","\\","beta$ param, default: 0.0, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"beta")),(0,r.kt)("li",{parentName:"ol"},"$\\lambda_1$: L1 regulation, default: 0.0, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"lambda1")),(0,r.kt)("li",{parentName:"ol"},"$\\lambda_2$: L2 regulation, default: 0.0, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"lambda2"))),(0,r.kt)("h2",{id:"adagrad"},"Adagrad"),(0,r.kt)("p",null,(0,r.kt)("a",{parentName:"p",href:"https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad"},"Adagrad")," configure the following parameters:"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},"$\\gamma$: learning rate, default: 1e-2, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"gamma")),(0,r.kt)("li",{parentName:"ol"},"$\\lambda$: weight decay, default: 0.0, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"lambda")),(0,r.kt)("li",{parentName:"ol"},"$\\eta$: learning rate decay, default: 0.0, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"eta")),(0,r.kt)("li",{parentName:"ol"},"$\\epsilon$: minimun error term, default: 1e-10, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"epsilon"))),(0,r.kt)("h2",{id:"adam"},"Adam"),(0,r.kt)("p",null,(0,r.kt)("a",{parentName:"p",href:"https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam"},"Adam")," configure the following parameters(not support amsgrad):"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},"$\\gamma$: learning rate, default: 1e-3, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"gamma")),(0,r.kt)("li",{parentName:"ol"},"$\\beta_1$: moving averages of gradient coefficient, default: 0.9, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"beta1")),(0,r.kt)("li",{parentName:"ol"},"$\\beta_2$: moving averages of gradient's square coefficient, default: 0.999, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"beta2")),(0,r.kt)("li",{parentName:"ol"},"$\\lambda$: weight decay rate, default: 0.0, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"lambda")),(0,r.kt)("li",{parentName:"ol"},"$\\epsilon$: minimun error term, default: 1e-8, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"epsilon"))),(0,r.kt)("h2",{id:"adamw"},"AdamW"),(0,r.kt)("p",null,(0,r.kt)("a",{parentName:"p",href:"https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW"},"AdamW")," configure the following parameters(not support amsgrad):"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},"$\\gamma$: learning rate, default: 1e-3, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"gamma")),(0,r.kt)("li",{parentName:"ol"},"$\\beta_1$: moving averages of gradient coefficient, default: 0.9, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"beta1")),(0,r.kt)("li",{parentName:"ol"},"$\\beta_2$: moving averages of gradient's square coefficient, default: 0.999, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"beta2")),(0,r.kt)("li",{parentName:"ol"},"$\\lambda$: weight decay rate, default: 1e-2, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"lambda")),(0,r.kt)("li",{parentName:"ol"},"$\\epsilon$: minimun error term, default: 1e-8, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"epsilon"))),(0,r.kt)("h2",{id:"lion"},"Lion"),(0,r.kt)("p",null,(0,r.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/2302.06675"},"Lion")," configure the following parameters:"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},"$\\eta$: learing rate, default: 3e-4, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"eta")),(0,r.kt)("li",{parentName:"ol"},"$\\beta_1$: moving averages of gradient coefficient, default: 0.9, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"beta1")),(0,r.kt)("li",{parentName:"ol"},"$\\beta_2$: moving averages of gradient's square coefficient, default: 0.99, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"beta2")),(0,r.kt)("li",{parentName:"ol"},"$\\lambda$: weight decay, default: 0.01, ",(0,r.kt)("strong",{parentName:"li"},"configure key"),": ",(0,r.kt)("inlineCode",{parentName:"li"},"lambda"))),(0,r.kt)("h2",{id:"example"},"Example"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},'import damo\nimport numpy as np\n\n# configure learning rate scheduler\nschedluer_params = damo.Parameters()\nschedluer_params.insert("name": "")\n\n# configure optimizer\noptimizer_params = damo.Parameters()\noptimizer_params.insert("name": "sgd")\noptimizer_params.insert("gamma": 0.001)\noptimizer_params.insert("lambda": 0.0)\n\n# no scheduler\nopt1 = damo.PyOptimizer(optimizer_params)\n\n# specific scheduler\nopt1 = damo.PyOptimizer(optimizer_params, schedluer_params)\n\nw = np.zeros(10, dtype=np.float32)\ngs = np.random.random(10).astype(np.float32)\nstep = 0\nopt1.call(w, gs, step)\n')))}g.isMDXComponent=!0}}]);