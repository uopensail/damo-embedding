"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[333],{3905:(e,n,t)=>{t.d(n,{Zo:()=>d,kt:()=>c});var r=t(7294);function a(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function o(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function i(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?o(Object(t),!0).forEach((function(n){a(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):o(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function s(e,n){if(null==e)return{};var t,r,a=function(e,n){if(null==e)return{};var t,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)t=o[r],n.indexOf(t)>=0||(a[t]=e[t]);return a}(e,n);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)t=o[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(a[t]=e[t])}return a}var l=r.createContext({}),m=function(e){var n=r.useContext(l),t=n;return e&&(t="function"==typeof e?e(n):i(i({},n),e)),t},d=function(e){var n=m(e.components);return r.createElement(l.Provider,{value:n},e.children)},p="mdxType",f={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},u=r.forwardRef((function(e,n){var t=e.components,a=e.mdxType,o=e.originalType,l=e.parentName,d=s(e,["components","mdxType","originalType","parentName"]),p=m(t),u=a,c=p["".concat(l,".").concat(u)]||p[u]||f[u]||o;return t?r.createElement(c,i(i({ref:n},d),{},{components:t})):r.createElement(c,i({ref:n},d))}));function c(e,n){var t=arguments,a=n&&n.mdxType;if("string"==typeof e||a){var o=t.length,i=new Array(o);i[0]=u;var s={};for(var l in n)hasOwnProperty.call(n,l)&&(s[l]=n[l]);s.originalType=e,s[p]="string"==typeof e?e:a,i[1]=s;for(var m=2;m<o;m++)i[m]=t[m];return r.createElement.apply(null,i)}return r.createElement.apply(null,t)}u.displayName="MDXCreateElement"},8125:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>l,contentTitle:()=>i,default:()=>f,frontMatter:()=>o,metadata:()=>s,toc:()=>m});var r=t(7462),a=(t(7294),t(3905));const o={},i="Example",s={unversionedId:"Example",id:"Example",title:"Example",description:"DeepFM",source:"@site/docs/Example.md",sourceDirName:".",slug:"/Example",permalink:"/damo-embedding/zh-Hans/docs/Example",draft:!1,editUrl:"https://github.com/uopensail/damo-embedding/edit/docs/docs/docs/Example.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",next:{title:"Damo-Embedding",permalink:"/damo-embedding/zh-Hans/docs/Intro"}},l={},m=[{value:"DeepFM",id:"deepfm",level:2},{value:"Save Model",id:"save-model",level:2}],d={toc:m},p="wrapper";function f(e){let{components:n,...t}=e;return(0,a.kt)(p,(0,r.Z)({},d,t,{components:n,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"example"},"Example"),(0,a.kt)("h2",{id:"deepfm"},"DeepFM"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'import torch\nimport torch.nn as nn\n\nfrom damo_embedding import Embedding\n\n\nclass DeepFM(torch.nn.Module):\n    def __init__(\n        self,\n        emb_size: int,\n        fea_size: int,\n        hid_dims=[256, 128],\n        num_classes=1,\n        dropout=[0.2, 0.2],\n        **kwargs,\n    ):\n        super(DeepFM, self).__init__()\n        self.emb_size = emb_size\n        self.fea_size = fea_size\n\n        initializer = {\n            "name": "truncate_normal",\n            "mean": float(kwargs.get("mean", 0.0)),\n            "stddev": float(kwargs.get("stddev", 0.0001)),\n        }\n\n        optimizer = {\n            "name": "adam",\n            "gamma": float(kwargs.get("gamma", 0.001)),\n            "beta1": float(kwargs.get("beta1", 0.9)),\n            "beta2": float(kwargs.get("beta2", 0.999)),\n            "lambda": float(kwargs.get("lambda", 0.0)),\n            "epsilon": float(kwargs.get("epsilon", 1e-8)),\n        }\n\n        self.w = Embedding(\n            1,\n            initializer=initializer,\n            optimizer=optimizer,\n            **kwargs,\n        )\n\n        self.v = Embedding(\n            self.emb_size,\n            initializer=initializer,\n            optimizer=optimizer,\n            **kwargs,\n        )\n        self.w0 = torch.zeros(1, dtype=torch.float32, requires_grad=True)\n        self.dims = [fea_size * emb_size] + hid_dims\n\n        self.layers = nn.ModuleList()\n        for i in range(1, len(self.dims)):\n            self.layers.append(nn.Linear(self.dims[i - 1], self.dims[i]))\n            self.layers.append(nn.BatchNorm1d(self.dims[i]))\n            self.layers.append(nn.BatchNorm1d(self.dims[i]))\n            self.layers.append(nn.ReLU())\n            self.layers.append(nn.Dropout(dropout[i - 1]))\n        self.layers.append(nn.Linear(self.dims[-1], num_classes))\n        self.sigmoid = nn.Sigmoid()\n\n    def forward(self, input: torch.Tensor) -> torch.Tensor:\n        """forward\n\n        Args:\n            input (torch.Tensor): input tensor\n\n        Returns:\n            tensor.Tensor: deepfm forward values\n        """\n        assert input.shape[1] == self.fea_size\n        w = self.w.forward(input)\n        v = self.v.forward(input)\n        square_of_sum = torch.pow(torch.sum(v, dim=1), 2)\n        sum_of_square = torch.sum(v * v, dim=1)\n        fm_out = (\n            torch.sum((square_of_sum - sum_of_square)\n                      * 0.5, dim=1, keepdim=True)\n            + torch.sum(w, dim=1)\n            + self.w0\n        )\n\n        dnn_out = torch.flatten(v, 1)\n        for layer in self.layers:\n            dnn_out = layer(dnn_out)\n        out = fm_out + dnn_out\n        out = self.sigmoid(out)\n        return out\n\n')),(0,a.kt)("h2",{id:"save-model"},"Save Model"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'from damo_embedding import save_model\nmodel = DeepFM(8, 39)\nsave_model(model, "./")\n')))}f.isMDXComponent=!0}}]);