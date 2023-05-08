"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[605],{3905:(e,t,r)=>{r.d(t,{Zo:()=>p,kt:()=>f});var n=r(7294);function a(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function o(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?o(Object(r),!0).forEach((function(t){a(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function l(e,t){if(null==e)return{};var r,n,a=function(e,t){if(null==e)return{};var r,n,a={},o=Object.keys(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||(a[r]=e[r]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var s=n.createContext({}),c=function(e){var t=n.useContext(s),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},p=function(e){var t=c(e.components);return n.createElement(s.Provider,{value:t},e.children)},u="mdxType",d={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},m=n.forwardRef((function(e,t){var r=e.components,a=e.mdxType,o=e.originalType,s=e.parentName,p=l(e,["components","mdxType","originalType","parentName"]),u=c(r),m=a,f=u["".concat(s,".").concat(m)]||u[m]||d[m]||o;return r?n.createElement(f,i(i({ref:t},p),{},{components:r})):n.createElement(f,i({ref:t},p))}));function f(e,t){var r=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=r.length,i=new Array(o);i[0]=m;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l[u]="string"==typeof e?e:a,i[1]=l;for(var c=2;c<o;c++)i[c]=r[c];return n.createElement.apply(null,i)}return n.createElement.apply(null,r)}m.displayName="MDXCreateElement"},9751:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>s,contentTitle:()=>i,default:()=>d,frontMatter:()=>o,metadata:()=>l,toc:()=>c});var n=r(7462),a=(r(7294),r(3905));const o={},i="Counting Bloom Filter",l={unversionedId:"modules/CBF",id:"modules/CBF",title:"Counting Bloom Filter",description:"Why Counting Bloom Filter",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/modules/CBF.md",sourceDirName:"modules",slug:"/modules/CBF",permalink:"/damo-embedding/zh-Hans/docs/modules/CBF",draft:!1,editUrl:"https://github.com/uopensail/damo-embedding/edit/docs/docs/docs/modules/CBF.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Modules",permalink:"/damo-embedding/zh-Hans/docs/modules/"},next:{title:"Embedding",permalink:"/damo-embedding/zh-Hans/docs/modules/Embedding"}},s={},c=[{value:"Why Counting Bloom Filter",id:"why-counting-bloom-filter",level:2},{value:"Configuration",id:"configuration",level:2},{value:"Example",id:"example",level:2},{value:"Reference",id:"reference",level:2}],p={toc:c},u="wrapper";function d(e){let{components:t,...r}=e;return(0,a.kt)(u,(0,n.Z)({},p,r,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"counting-bloom-filter"},"Counting Bloom Filter"),(0,a.kt)("h2",{id:"why-counting-bloom-filter"},"Why Counting Bloom Filter"),(0,a.kt)("p",null,"The purpose of the Counting Bloom Filter(abbr. CBF) is to filter low-frequency features. "),(0,a.kt)("p",null,"Long-tail is a distinctive characteristic in internet scenario. When training machine learning/deep learning models, there will be too many long-tail features, its requency is quite low. They do harm to the convergence of the model: on the one hand, the low-frequency features are not fully trained, on the other hand, they also waste large storage resources and computing resources, so it is very necessary to remove low-frequency features."),(0,a.kt)("p",null,"If the model is trained offline, the engineer can preprocess the features, count the frequency of each feature, and then remove these low-frequency features. However, if it is an online model, it is not possible to preprocess the features. There are many schemes for processing sparse features, such as: feature frequency estimation based on Poisson distribution, dynamic adjustment of L1 regular filtering, etc.","[1]",". "),(0,a.kt)("p",null,"We provides a relatively straightforward way, using the CBF to record the number of feature's frequency. It should be noted that we uses ",(0,a.kt)("inlineCode",{parentName:"p"},"4bit"),"to store the number, which means that the maximum frequency is 15. Because we believe that the value of 15 can meet most of the needs."),(0,a.kt)("p",null,"Also, to avoid the problem of data loss. We use ",(0,a.kt)("inlineCode",{parentName:"p"},"mmap"),",which maps the file to memory, to save data. If the model training finishes or crashes, data has already been saved to the disk. When the model training is restarted, you can reload the data from disk"),(0,a.kt)("p",null,"There is a question on stackoverflow, ",(0,a.kt)("a",{parentName:"p",href:"https://stackoverflow.com/questions/44815329/what-updates-mtime-after-writing-to-memory-mapped-files"},"What updates mtime after writing to memory mapped files?")),(0,a.kt)("blockquote",null,(0,a.kt)("p",{parentName:"blockquote"},"When you\xa0",(0,a.kt)("inlineCode",{parentName:"p"},"mmap"),"\xa0a file, you're basically sharing memory directly between your process and the kernel's page cache \u2014 the same cache that holds file data that's been read from disk, or is waiting to be written to disk. A page in the page cache that's different from what's on disk (because it's been written to) is referred to as \"dirty\".\nThere is a kernel thread that scans for dirty pages and writes them back to disk, under the control of several parameters. One important one is\xa0",(0,a.kt)("inlineCode",{parentName:"p"},"dirty_expire_centisecs"),". If any of the pages for a file have been dirty for longer than\xa0",(0,a.kt)("inlineCode",{parentName:"p"},"dirty_expire_centisecs"),"\xa0then all of the dirty pages for that file will get written out. The default value is 3000 centisecs (30 seconds).")),(0,a.kt)("p",null,"Because mmap writes data to disk periodically, there is no need to create a new thread to write data to disk."),(0,a.kt)("h2",{id:"configuration"},"Configuration"),(0,a.kt)("ol",null,(0,a.kt)("li",{parentName:"ol"},(0,a.kt)("p",{parentName:"li"},"capacity: max capacity of CBF, default: $2^{28}$")),(0,a.kt)("li",{parentName:"ol"},(0,a.kt)("p",{parentName:"li"},"count: filter count, default: 15")),(0,a.kt)("li",{parentName:"ol"},(0,a.kt)("p",{parentName:"li"},"path: data path, default: /tmp/COUNTING_BLOOM_FILTER_DATA")),(0,a.kt)("li",{parentName:"ol"},(0,a.kt)("p",{parentName:"li"},"fpr: false positive rate, default: 1e-3")),(0,a.kt)("li",{parentName:"ol"},(0,a.kt)("p",{parentName:"li"},"reload: whether read data from disk file, defalut: true"))),(0,a.kt)("h2",{id:"example"},"Example"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'import damo\n\nparam = damo.Parameters()\nparam.insert("capacity", 1<<28) \nparam.insert("count", 15)  \nparam.insert("path", "/tmp/cbf")\nparam.insert("reload", True)\nparam.insert("fpr", 0.001)\nprint(param.to_json())\n\nfilter = damo.PyFilter(param)\n\nkey = 123456\nfor i in range(16):\n    filter.add(key, 1)\n    print(filter.check(key))\n')),(0,a.kt)("h2",{id:"reference"},"Reference"),(0,a.kt)("p",null,"[1][Ant Financial's core technology: real-time recommendation algorithm for tens of billions of features]","(",(0,a.kt)("a",{parentName:"p",href:"https://developer.aliyun.com/article/714366"},"https://developer.aliyun.com/article/714366"),")"))}d.isMDXComponent=!0}}]);