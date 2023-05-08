"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[525],{3905:(e,t,a)=>{a.d(t,{Zo:()=>d,kt:()=>y});var n=a(7294);function r(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function l(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function o(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?l(Object(a),!0).forEach((function(t){r(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):l(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function i(e,t){if(null==e)return{};var a,n,r=function(e,t){if(null==e)return{};var a,n,r={},l=Object.keys(e);for(n=0;n<l.length;n++)a=l[n],t.indexOf(a)>=0||(r[a]=e[a]);return r}(e,t);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(n=0;n<l.length;n++)a=l[n],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(r[a]=e[a])}return r}var p=n.createContext({}),c=function(e){var t=n.useContext(p),a=t;return e&&(a="function"==typeof e?e(t):o(o({},t),e)),a},d=function(e){var t=c(e.components);return n.createElement(p.Provider,{value:t},e.children)},s="mdxType",m={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},u=n.forwardRef((function(e,t){var a=e.components,r=e.mdxType,l=e.originalType,p=e.parentName,d=i(e,["components","mdxType","originalType","parentName"]),s=c(a),u=r,y=s["".concat(p,".").concat(u)]||s[u]||m[u]||l;return a?n.createElement(y,o(o({ref:t},d),{},{components:a})):n.createElement(y,o({ref:t},d))}));function y(e,t){var a=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var l=a.length,o=new Array(l);o[0]=u;var i={};for(var p in t)hasOwnProperty.call(t,p)&&(i[p]=t[p]);i.originalType=e,i[s]="string"==typeof e?e:r,o[1]=i;for(var c=2;c<l;c++)o[c]=a[c];return n.createElement.apply(null,o)}return n.createElement.apply(null,a)}u.displayName="MDXCreateElement"},9012:(e,t,a)=>{a.r(t),a.d(t,{assets:()=>p,contentTitle:()=>o,default:()=>m,frontMatter:()=>l,metadata:()=>i,toc:()=>c});var n=a(7462),r=(a(7294),a(3905));const l={},o="Learning Rate Scheduler",i={unversionedId:"modules/Scheduler",id:"modules/Scheduler",title:"Learning Rate Scheduler",description:"When training deep neural networks, it is often useful to reduce learning rate as the training progresses. Learning rate schedules seek to adjust the learning rate during training by reducing the learning rate according to a pre-defined schedule.",source:"@site/docs/modules/Scheduler.md",sourceDirName:"modules",slug:"/modules/Scheduler",permalink:"/damo-embedding/docs/modules/Scheduler",draft:!1,editUrl:"https://github.com/uopensail/damo-embedding/edit/docs/docs/docs/modules/Scheduler.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Optimizer",permalink:"/damo-embedding/docs/modules/Optimizer"},next:{title:"Storage",permalink:"/damo-embedding/docs/modules/Storage"}},p={},c=[{value:"Exponential Decay",id:"exponential-decay",level:2},{value:"Polynomial Decay",id:"polynomial-decay",level:2},{value:"Nature Exponential Decay",id:"nature-exponential-decay",level:2},{value:"Inverse Time Decay",id:"inverse-time-decay",level:2},{value:"Cosine Decay",id:"cosine-decay",level:2},{value:"Liner Cosine Decay",id:"liner-cosine-decay",level:2}],d={toc:c},s="wrapper";function m(e){let{components:t,...a}=e;return(0,r.kt)(s,(0,n.Z)({},d,a,{components:t,mdxType:"MDXLayout"}),(0,r.kt)("h1",{id:"learning-rate-scheduler"},"Learning Rate Scheduler"),(0,r.kt)("p",null,"When training deep neural networks, it is often useful to reduce learning rate as the training progresses. Learning rate schedules seek to adjust the learning rate during training by reducing the learning rate according to a pre-defined schedule. "),(0,r.kt)("p",null,"When using an scheduler, you need to configure the ",(0,r.kt)("inlineCode",{parentName:"p"},"name")," item to indicate which scheduler to use, and then configure their respective parameters according to different scheduler. The following are the names of each scheduler. If name is empty, no scheduler is used."),(0,r.kt)("table",null,(0,r.kt)("thead",{parentName:"table"},(0,r.kt)("tr",{parentName:"thead"},(0,r.kt)("th",{parentName:"tr",align:"center"},"scheduler"),(0,r.kt)("th",{parentName:"tr",align:"center"},"name"))),(0,r.kt)("tbody",{parentName:"table"},(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"center"},"Exponential Decay"),(0,r.kt)("td",{parentName:"tr",align:"center"},"exponential_decay")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"center"},"Polynomial Decay"),(0,r.kt)("td",{parentName:"tr",align:"center"},"polynomial_decay")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"center"},"Nature Exponential Decay"),(0,r.kt)("td",{parentName:"tr",align:"center"},"nature_exponential_decay")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"center"},"Inverse Time Decay"),(0,r.kt)("td",{parentName:"tr",align:"center"},"inverse_time_decay")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"center"},"Cosine Decay"),(0,r.kt)("td",{parentName:"tr",align:"center"},"cosine_decay")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"center"},"Liner Cosine Decay"),(0,r.kt)("td",{parentName:"tr",align:"center"},"liner_cosine_decay")))),(0,r.kt)("h2",{id:"exponential-decay"},"Exponential Decay"),(0,r.kt)("p",null,"$learning","_","rate * decay","_","rate ^{\\frac{global","_","step}{decay","_","steps}}$"),(0,r.kt)("p",null,"configure the following parameters:"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("p",{parentName:"li"},(0,r.kt)("strong",{parentName:"p"},"decay_steps"),": float type")),(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("p",{parentName:"li"},(0,r.kt)("strong",{parentName:"p"},"decay_rate:")," float type"))),(0,r.kt)("h2",{id:"polynomial-decay"},"Polynomial Decay"),(0,r.kt)("p",null,"$(learning","_","rate - end","_","learning","_","rate)*decay","_","rate^{1.0 - \\frac{min(global","_","step, decay","_","steps)}{decay","_","steps}} + end","_","learning","_","rate$"),(0,r.kt)("p",null,"configure the following parameters:"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("p",{parentName:"li"},(0,r.kt)("strong",{parentName:"p"},"decay_steps"),": float type")),(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("p",{parentName:"li"},(0,r.kt)("strong",{parentName:"p"},"decay_rate"),": float type, default: 1e-3")),(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("p",{parentName:"li"},(0,r.kt)("strong",{parentName:"p"},"end_learning_rate"),": float type, default: 1.0"))),(0,r.kt)("h2",{id:"nature-exponential-decay"},"Nature Exponential Decay"),(0,r.kt)("p",null,"$learning","_","rate",(0,r.kt)("em",{parentName:"p"},"e^{-decay","_","rate "),"{\\frac{global","_","step}{decay","_","steps}}}$"),(0,r.kt)("p",null,"configure the following parameters:"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("p",{parentName:"li"},(0,r.kt)("strong",{parentName:"p"},"decay_steps"),": float type")),(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("p",{parentName:"li"},(0,r.kt)("strong",{parentName:"p"},"decay_rate:")," float type"))),(0,r.kt)("h2",{id:"inverse-time-decay"},"Inverse Time Decay"),(0,r.kt)("p",null,"$\\frac{learning","_","rate}{1.0+ decay","_","rate *{\\frac{global","_","step}{decay","_","steps}}}$"),(0,r.kt)("p",null,"configure the following parameters:"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("p",{parentName:"li"},(0,r.kt)("strong",{parentName:"p"},"decay_steps"),": float type")),(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("p",{parentName:"li"},(0,r.kt)("strong",{parentName:"p"},"decay_rate:")," float type"))),(0,r.kt)("h2",{id:"cosine-decay"},"Cosine Decay"),(0,r.kt)("p",null,"$learning","_","rate ",(0,r.kt)("em",{parentName:"p"}," 0.5 "),"(1.0 + cos(\\pi*\\frac{global","_","step}{decay","_","steps})$"),(0,r.kt)("p",null,"configure the following parameters:"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("strong",{parentName:"li"},"decay_steps"),": float type")),(0,r.kt)("h2",{id:"liner-cosine-decay"},"Liner Cosine Decay"),(0,r.kt)("p",null,"$liner","_","decay = \\frac{decay","_","steps - min(global","_","step, decay","_","steps)}{decay","_","steps}$"),(0,r.kt)("p",null,"$cos","_","decay = -0.5 ",(0,r.kt)("em",{parentName:"p"}," (1.0 + cos(2\\pi"),"num","_","periods*\\frac{min(global","_","step, decay","_","steps)}{decay","_","steps})$"),(0,r.kt)("p",null,"$learning","_","rate ",(0,r.kt)("em",{parentName:"p"}," (\\alpha + liner","_","decay)"),"cos","_","decay+\\beta$"),(0,r.kt)("p",null,"configure the following parameters:"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("p",{parentName:"li"},(0,r.kt)("strong",{parentName:"p"},"alpha"),": $\\alpha$, float type, default: 0.0")),(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("p",{parentName:"li"},(0,r.kt)("strong",{parentName:"p"},"beta"),": $\\beta$, float type, default: 1e-3")),(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("p",{parentName:"li"},(0,r.kt)("strong",{parentName:"p"},"num_periods"),": float type, default: 0.5")),(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("p",{parentName:"li"},(0,r.kt)("strong",{parentName:"p"},"decay_steps"),": float type"))))}m.isMDXComponent=!0}}]);