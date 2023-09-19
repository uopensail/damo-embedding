"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[706],{3905:(e,t,r)=>{r.d(t,{Zo:()=>m,kt:()=>f});var n=r(7294);function o(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function i(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function a(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?i(Object(r),!0).forEach((function(t){o(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):i(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function l(e,t){if(null==e)return{};var r,n,o=function(e,t){if(null==e)return{};var r,n,o={},i=Object.keys(e);for(n=0;n<i.length;n++)r=i[n],t.indexOf(r)>=0||(o[r]=e[r]);return o}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(n=0;n<i.length;n++)r=i[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var s=n.createContext({}),d=function(e){var t=n.useContext(s),r=t;return e&&(r="function"==typeof e?e(t):a(a({},t),e)),r},m=function(e){var t=d(e.components);return n.createElement(s.Provider,{value:t},e.children)},u="mdxType",p={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},c=n.forwardRef((function(e,t){var r=e.components,o=e.mdxType,i=e.originalType,s=e.parentName,m=l(e,["components","mdxType","originalType","parentName"]),u=d(r),c=o,f=u["".concat(s,".").concat(c)]||u[c]||p[c]||i;return r?n.createElement(f,a(a({ref:t},m),{},{components:r})):n.createElement(f,a({ref:t},m))}));function f(e,t){var r=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var i=r.length,a=new Array(i);a[0]=c;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l[u]="string"==typeof e?e:o,a[1]=l;for(var d=2;d<i;d++)a[d]=r[d];return n.createElement.apply(null,a)}return n.createElement.apply(null,r)}c.displayName="MDXCreateElement"},9959:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>s,contentTitle:()=>a,default:()=>p,frontMatter:()=>i,metadata:()=>l,toc:()=>d});var n=r(7462),o=(r(7294),r(3905));const i={},a=void 0,l={unversionedId:"modules/Modules",id:"modules/Modules",title:"Modules",description:"Modules",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/modules/Modules.md",sourceDirName:"modules",slug:"/modules/",permalink:"/damo-embedding/zh-Hans/docs/modules/",draft:!1,editUrl:"https://github.com/uopensail/damo-embedding/edit/docs/docs/docs/modules/Modules.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Rocksdb Install",permalink:"/damo-embedding/zh-Hans/docs/install/RocksDB"},next:{title:"Counting Bloom Filter",permalink:"/damo-embedding/zh-Hans/docs/modules/CBF"}},s={},d=[{value:"Modules",id:"modules",level:2},{value:"Initializer",id:"initializer",level:3},{value:"Optimizer",id:"optimizer",level:3},{value:"Storage",id:"storage",level:3},{value:"Embedding",id:"embedding",level:3}],m={toc:d},u="wrapper";function p(e){let{components:t,...r}=e;return(0,o.kt)(u,(0,n.Z)({},m,r,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h2",{id:"modules"},"Modules"),(0,o.kt)("p",null,"Counting Bloom Filter"),(0,o.kt)("p",null,"The purpose of the Counting Bloom Filter is to filter low-frequency features. For more details, please refer to ",(0,o.kt)("a",{parentName:"p",href:"/damo-embedding/zh-Hans/docs/modules/CBF"},"Counting Bloom Filter"),"."),(0,o.kt)("h3",{id:"initializer"},"Initializer"),(0,o.kt)("p",null,"When user lookup keys from the embedding, if a key not exists, we use a specific initializer to initialize its weight, and then send the weight to the user, also save the weight to rocksdb. For more detail, please refer to ",(0,o.kt)("a",{parentName:"p",href:"/damo-embedding/zh-Hans/docs/modules/Initializer"},"Initializer"),"."),(0,o.kt)("h3",{id:"optimizer"},"Optimizer"),(0,o.kt)("p",null,"There are many different optimizers, user can pick one of them, and use this optimizer to apply gradients. For more detail, please refer to ",(0,o.kt)("a",{parentName:"p",href:"/damo-embedding/zh-Hans/docs/modules/Optimizer"},"Optimizer"),"."),(0,o.kt)("h3",{id:"storage"},"Storage"),(0,o.kt)("p",null,"The storage is based on rocksdb, it supports TTL(Time To Live). When creating a embedding object, storage object is necessary. Also, we support dump data to binary file for online serving. For more detail, please refer to ",(0,o.kt)("a",{parentName:"p",href:"/damo-embedding/zh-Hans/docs/modules/Storage"},"Storage"),"."),(0,o.kt)("h3",{id:"embedding"},"Embedding"),(0,o.kt)("p",null,"This is the most important module in this project. When creating an embedding object, users need fill in 5 arguments listed below:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"storage"),": damo.PyStorage type"),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"optimizer"),": damo.PyOptimizer type"),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"initializer"),": damo.PyInitializer type"),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"dimension"),": int type, dim of embedding"),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"group"),": int type, [0, 256), defaul: 0")),(0,o.kt)("p",null,"Embedding moule has two member functions: lookup and apply_gradients, both have no return values. For more detail, please refer to ",(0,o.kt)("a",{parentName:"p",href:"/damo-embedding/zh-Hans/docs/modules/Embedding"},"Embedding"),"."))}p.isMDXComponent=!0}}]);