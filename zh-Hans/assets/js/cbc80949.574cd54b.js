"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[321],{3905:(t,e,n)=>{n.d(e,{Zo:()=>u,kt:()=>g});var r=n(7294);function a(t,e,n){return e in t?Object.defineProperty(t,e,{value:n,enumerable:!0,configurable:!0,writable:!0}):t[e]=n,t}function l(t,e){var n=Object.keys(t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(t);e&&(r=r.filter((function(e){return Object.getOwnPropertyDescriptor(t,e).enumerable}))),n.push.apply(n,r)}return n}function o(t){for(var e=1;e<arguments.length;e++){var n=null!=arguments[e]?arguments[e]:{};e%2?l(Object(n),!0).forEach((function(e){a(t,e,n[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(n)):l(Object(n)).forEach((function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(n,e))}))}return t}function i(t,e){if(null==t)return{};var n,r,a=function(t,e){if(null==t)return{};var n,r,a={},l=Object.keys(t);for(r=0;r<l.length;r++)n=l[r],e.indexOf(n)>=0||(a[n]=t[n]);return a}(t,e);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(t);for(r=0;r<l.length;r++)n=l[r],e.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(t,n)&&(a[n]=t[n])}return a}var p=r.createContext({}),d=function(t){var e=r.useContext(p),n=e;return t&&(n="function"==typeof t?t(e):o(o({},e),t)),n},u=function(t){var e=d(t.components);return r.createElement(p.Provider,{value:e},t.children)},s="mdxType",m={inlineCode:"code",wrapper:function(t){var e=t.children;return r.createElement(r.Fragment,{},e)}},c=r.forwardRef((function(t,e){var n=t.components,a=t.mdxType,l=t.originalType,p=t.parentName,u=i(t,["components","mdxType","originalType","parentName"]),s=d(n),c=a,g=s["".concat(p,".").concat(c)]||s[c]||m[c]||l;return n?r.createElement(g,o(o({ref:e},u),{},{components:n})):r.createElement(g,o({ref:e},u))}));function g(t,e){var n=arguments,a=e&&e.mdxType;if("string"==typeof t||a){var l=n.length,o=new Array(l);o[0]=c;var i={};for(var p in e)hasOwnProperty.call(e,p)&&(i[p]=e[p]);i.originalType=t,i[s]="string"==typeof t?t:a,o[1]=i;for(var d=2;d<l;d++)o[d]=n[d];return r.createElement.apply(null,o)}return r.createElement.apply(null,n)}c.displayName="MDXCreateElement"},1895:(t,e,n)=>{n.r(e),n.d(e,{assets:()=>p,contentTitle:()=>o,default:()=>m,frontMatter:()=>l,metadata:()=>i,toc:()=>d});var r=n(7462),a=(n(7294),n(3905));const l={},o="Storage",i={unversionedId:"modules/Storage",id:"modules/Storage",title:"Storage",description:"We use rocksdb to save the data. You should create the storage object fisrt, then you can create embedding objects.",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/modules/Storage.md",sourceDirName:"modules",slug:"/modules/Storage",permalink:"/damo-embedding/zh-Hans/docs/modules/Storage",draft:!1,editUrl:"https://github.com/uopensail/damo-embedding/edit/docs/docs/docs/modules/Storage.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Learning Rate Scheduler",permalink:"/damo-embedding/zh-Hans/docs/modules/Scheduler"},next:{title:"\u5feb\u901f\u5b89\u88c5",permalink:"/damo-embedding/zh-Hans/docs/quick_install"}},p={},d=[{value:"condition",id:"condition",level:2},{value:"expire_days",id:"expire_days",level:3},{value:"group",id:"group",level:3},{value:"min_count",id:"min_count",level:3},{value:"dump",id:"dump",level:2},{value:"file format",id:"file-format",level:3},{value:"first part",id:"first-part",level:4},{value:"second part",id:"second-part",level:4},{value:"Example",id:"example",level:2}],u={toc:d},s="wrapper";function m(t){let{components:e,...n}=t;return(0,a.kt)(s,(0,r.Z)({},u,n,{components:e,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"storage"},"Storage"),(0,a.kt)("p",null,"We use rocksdb to save the data. You should create the storage object fisrt, then you can create embedding objects."),(0,a.kt)("p",null,"When creating storage object, you should input data-dir and ttl. Data-dir is the path to save data. Ttl is time to live, which is supported by rocksdb."),(0,a.kt)("h2",{id:"condition"},"condition"),(0,a.kt)("p",null,"conditions are wrap in parameter object."),(0,a.kt)("h3",{id:"expire_days"},"expire_days"),(0,a.kt)("p",null,"If the last update time of the key is less then expire_days ago, this key is ignored."),(0,a.kt)("p",null,"configure parameters: "),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"expire_days"),": int type"),(0,a.kt)("h3",{id:"group"},"group"),(0,a.kt)("p",null,"If the group is setted (0 <= group < 256),  we will only dump keys have the same group."),(0,a.kt)("p",null,"configure parameters:"),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"group"),": int type"),(0,a.kt)("h3",{id:"min_count"},"min_count"),(0,a.kt)("p",null,"If the update number of key is less then min_count, this key is ignored."),(0,a.kt)("p",null,"configure parameters:"),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"min_count"),": int type"),(0,a.kt)("h2",{id:"dump"},"dump"),(0,a.kt)("p",null,"When model training finishes, you may dump the weights of keys with some conditions."),(0,a.kt)("h3",{id:"file-format"},"file format"),(0,a.kt)("p",null,"First part stores the dim and count of each group, totally 256 groups. Second part stores all the key and weigh."),(0,a.kt)("h4",{id:"first-part"},"first part"),(0,a.kt)("table",null,(0,a.kt)("thead",{parentName:"table"},(0,a.kt)("tr",{parentName:"thead"},(0,a.kt)("th",{parentName:"tr",align:null},"type"),(0,a.kt)("th",{parentName:"tr",align:null},"size"),(0,a.kt)("th",{parentName:"tr",align:null},"length"),(0,a.kt)("th",{parentName:"tr",align:null},"description"))),(0,a.kt)("tbody",{parentName:"table"},(0,a.kt)("tr",{parentName:"tbody"},(0,a.kt)("td",{parentName:"tr",align:null},"int32"),(0,a.kt)("td",{parentName:"tr",align:null},"4bit"),(0,a.kt)("td",{parentName:"tr",align:null},"256"),(0,a.kt)("td",{parentName:"tr",align:null},"dim of 256 group, default 0")),(0,a.kt)("tr",{parentName:"tbody"},(0,a.kt)("td",{parentName:"tr",align:null},"size_t"),(0,a.kt)("td",{parentName:"tr",align:null},"8bit"),(0,a.kt)("td",{parentName:"tr",align:null},"256"),(0,a.kt)("td",{parentName:"tr",align:null},"key count of 256 group, default 0")))),(0,a.kt)("h4",{id:"second-part"},"second part"),(0,a.kt)("table",null,(0,a.kt)("thead",{parentName:"table"},(0,a.kt)("tr",{parentName:"thead"},(0,a.kt)("th",{parentName:"tr",align:null},"type"),(0,a.kt)("th",{parentName:"tr",align:null},"size"),(0,a.kt)("th",{parentName:"tr",align:null},"length"),(0,a.kt)("th",{parentName:"tr",align:null},"description"))),(0,a.kt)("tbody",{parentName:"table"},(0,a.kt)("tr",{parentName:"tbody"},(0,a.kt)("td",{parentName:"tr",align:null},"u_int64_t"),(0,a.kt)("td",{parentName:"tr",align:null},"8bit"),(0,a.kt)("td",{parentName:"tr",align:null},"1"),(0,a.kt)("td",{parentName:"tr",align:null},"key value")),(0,a.kt)("tr",{parentName:"tbody"},(0,a.kt)("td",{parentName:"tr",align:null},"int32"),(0,a.kt)("td",{parentName:"tr",align:null},"4bit"),(0,a.kt)("td",{parentName:"tr",align:null},"1"),(0,a.kt)("td",{parentName:"tr",align:null},"group of the key")),(0,a.kt)("tr",{parentName:"tbody"},(0,a.kt)("td",{parentName:"tr",align:null},"float"),(0,a.kt)("td",{parentName:"tr",align:null},"4bit"),(0,a.kt)("td",{parentName:"tr",align:null},"dim of this group"),(0,a.kt)("td",{parentName:"tr",align:null},"weight of the key")))),(0,a.kt)("h2",{id:"example"},"Example"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'import damo\n\n# first param: data dir\n# second param: ttl second\nstorage = damo.PyStorage("/tmp/data_dir", 86400*100)\n\n\ncond = damo.Parameters()\ncond.insert("expire_days", 100)\ncond.insert("min_count", 3)\ncond.insert("group", 0)\n\nstorage.dump("/tmp/weight.dat", cond)\n\n')))}m.isMDXComponent=!0}}]);