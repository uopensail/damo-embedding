"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[185],{3905:(e,t,n)=>{n.d(t,{Zo:()=>s,kt:()=>y});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var u=r.createContext({}),p=function(e){var t=r.useContext(u),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},s=function(e){var t=p(e.components);return r.createElement(u.Provider,{value:t},e.children)},c="mdxType",m={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,o=e.originalType,u=e.parentName,s=l(e,["components","mdxType","originalType","parentName"]),c=p(n),d=a,y=c["".concat(u,".").concat(d)]||c[d]||m[d]||o;return n?r.createElement(y,i(i({ref:t},s),{},{components:n})):r.createElement(y,i({ref:t},s))}));function y(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=n.length,i=new Array(o);i[0]=d;var l={};for(var u in t)hasOwnProperty.call(t,u)&&(l[u]=t[u]);l.originalType=e,l[c]="string"==typeof e?e:a,i[1]=l;for(var p=2;p<o;p++)i[p]=n[p];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}d.displayName="MDXCreateElement"},6606:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>u,contentTitle:()=>i,default:()=>m,frontMatter:()=>o,metadata:()=>l,toc:()=>p});var r=n(7462),a=(n(7294),n(3905));const o={},i="SWIG and NumPy",l={unversionedId:"install/Swig&NumPy",id:"install/Swig&NumPy",title:"SWIG and NumPy",description:"SWIG\u5728\u8be5\u9879\u76ee\u4e2d\u4e3b\u8981\u662f\u5c06c++\u4ee3\u7801\u5c01\u88c5\u6210Python\u7684\u5de5\u5177\uff0c\u5177\u4f53\u7684\u5b89\u88c5\u53ef\u4ee5\u53c2\u8003\u8fd9\u7bc7\u7f51\u9875\uff1a",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/install/Swig&NumPy.md",sourceDirName:"install",slug:"/install/Swig&NumPy",permalink:"/damo-embedding/zh-Hans/docs/install/Swig&NumPy",draft:!1,editUrl:"https://github.com/uopensail/damo-embedding/edit/docs/docs/docs/install/Swig&NumPy.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Rocksdb Install",permalink:"/damo-embedding/zh-Hans/docs/install/RocksDB"},next:{title:"Modules",permalink:"/damo-embedding/zh-Hans/docs/modules/"}},u={},p=[{value:"\u8bed\u6cd5",id:"\u8bed\u6cd5",level:2},{value:"NumPy",id:"numpy",level:2},{value:"NumPy Add to System Path",id:"numpy-add-to-system-path",level:3}],s={toc:p},c="wrapper";function m(e){let{components:t,...n}=e;return(0,a.kt)(c,(0,r.Z)({},s,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"swig-and-numpy"},"SWIG and NumPy"),(0,a.kt)("p",null,"SWIG\u5728\u8be5\u9879\u76ee\u4e2d\u4e3b\u8981\u662f\u5c06c++\u4ee3\u7801\u5c01\u88c5\u6210Python\u7684\u5de5\u5177\uff0c\u5177\u4f53\u7684\u5b89\u88c5\u53ef\u4ee5\u53c2\u8003\u8fd9\u7bc7\u7f51\u9875\uff1a"),(0,a.kt)("p",null,(0,a.kt)("a",{parentName:"p",href:"https://www.dev2qa.com/how-to-install-swig-on-macos-linux-and-windows/"},"How To Install Swig On MacOS, Linux And Windows")),(0,a.kt)("h2",{id:"\u8bed\u6cd5"},"\u8bed\u6cd5"),(0,a.kt)("p",null,"\u5177\u4f53\u7684SWIG\u8bed\u6cd5\u53ef\u4ee5\u53c2\u8003\uff1a",(0,a.kt)("a",{parentName:"p",href:"https://www.swig.org/doc.html"},"SWIG\u5b98\u65b9\u6587\u6863")),(0,a.kt)("h2",{id:"numpy"},"NumPy"),(0,a.kt)("p",null,"\u5728\u8be5\u9879\u76ee\u4e2d\uff0c\u4f7f\u7528\u4e86numpy\u5e93\uff0c\u9700\u8981\u5c06SWIG\u548cnumpy\u7ed3\u5408\u8d77\u6765\u4f7f\u7528\uff0c\u6240\u4ee5\u9700\u8981\u63d0\u4f9b",(0,a.kt)("a",{parentName:"p",href:"https://github.com/numpy/numpy/blob/main/tools/swig/numpy.i"},"numyp.i"),"\u6587\u4ef6\uff0c\u5177\u4f53\u4f7f\u7528\u65b9\u6cd5\u53c2\u8003",(0,a.kt)("a",{parentName:"p",href:"https://numpy.org/doc/stable/reference/swig.interface-file.html"},"numpy.i: a SWIG Interface File for NumPy"),"\u3002"),(0,a.kt)("h3",{id:"numpy-add-to-system-path"},"NumPy Add to System Path"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-bash"},"# PYTHONPATH May Be Different\n\nPYTHONPATH=/usr/local/python3/lib/python3.7\nNUMPY_INCLUDE_PATH=$PYTHONPATH/site-packages/numpy/core/include\nNUMPY_LIBRARY_PATH=$PYTHONPATH/site-packages/numpy/core/lib\n\nexport CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$NUMPY_INCLUDE_PATH\nexport LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NUMPY_LIBRARY_PATH\nexport LIBRARY_PATH=$LIBRARY_PATH:NUMPY_LIBRARY_PATH\n")))}m.isMDXComponent=!0}}]);