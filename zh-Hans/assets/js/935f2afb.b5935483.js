"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[53],{1109:e=>{e.exports=JSON.parse('{"pluginId":"default","version":"current","label":"Next","banner":null,"badge":false,"noIndex":false,"className":"docs-version-current","isLast":true,"docsSidebars":{"tutorialSidebar":[{"type":"link","label":"Damo-Embedding","href":"/damo-embedding/zh-Hans/docs/Intro","docId":"Intro"},{"type":"category","label":"Install","collapsible":true,"collapsed":true,"items":[{"type":"link","label":"Python3 Install","href":"/damo-embedding/zh-Hans/docs/install/Python3","docId":"install/Python3"},{"type":"link","label":"RocksDB Install","href":"/damo-embedding/zh-Hans/docs/install/RocksDB","docId":"install/RocksDB"},{"type":"link","label":"SWIG and NumPy","href":"/damo-embedding/zh-Hans/docs/install/Swig&NumPy","docId":"install/Swig&NumPy"}],"href":"/damo-embedding/zh-Hans/docs/install/"},{"type":"category","label":"Modules","collapsible":true,"collapsed":true,"items":[{"type":"link","label":"Counting Bloom Filter","href":"/damo-embedding/zh-Hans/docs/modules/CBF","docId":"modules/CBF"},{"type":"link","label":"Embedding","href":"/damo-embedding/zh-Hans/docs/modules/Embedding","docId":"modules/Embedding"},{"type":"link","label":"Initializer","href":"/damo-embedding/zh-Hans/docs/modules/Initializer","docId":"modules/Initializer"},{"type":"link","label":"Optimizer","href":"/damo-embedding/zh-Hans/docs/modules/Optimizer","docId":"modules/Optimizer"},{"type":"link","label":"Learning Rate Scheduler","href":"/damo-embedding/zh-Hans/docs/modules/Scheduler","docId":"modules/Scheduler"},{"type":"link","label":"Storage","href":"/damo-embedding/zh-Hans/docs/modules/Storage","docId":"modules/Storage"}],"href":"/damo-embedding/zh-Hans/docs/modules/"},{"type":"link","label":"\u5feb\u901f\u5b89\u88c5","href":"/damo-embedding/zh-Hans/docs/quick_install","docId":"quick_install"}]},"docs":{"install/Install":{"id":"install/Install","title":"Install","description":"Install","sidebar":"tutorialSidebar"},"install/Python3":{"id":"install/Python3","title":"Python3 Install","description":"MacOS","sidebar":"tutorialSidebar"},"install/RocksDB":{"id":"install/RocksDB","title":"RocksDB Install","description":"Refer to rocksdb/INSTALL.md at master \xb7 facebook/rocksdb \xb7 GitHub\u3002Below are some easy install commands.","sidebar":"tutorialSidebar"},"install/Swig&NumPy":{"id":"install/Swig&NumPy","title":"SWIG and NumPy","description":"We use SWIG in this project to encapsulate C++ code into Python, SWIG installation can refer to this web page:","sidebar":"tutorialSidebar"},"Intro":{"id":"Intro","title":"Damo-Embedding","description":"\u8be5\u9879\u76ee\u4e3b\u8981\u9488\u5bf9\u7684\u662f\u5c0f\u516c\u53f8\u7684\u6a21\u578b\u8bad\u7ec3\u573a\u666f, \u56e0\u4e3a\u5c0f\u516c\u53f8\u5728\u673a\u5668\u8d44\u6e90\u65b9\u9762\u53ef\u80fd\u6bd4\u8f83\u53d7\u9650, \u4e0d\u592a\u5bb9\u6613\u7533\u8bf7\u5927\u5185\u5b58\u7684\u673a\u5668\u4ee5\u53ca\u5206\u5e03\u5f0f\u7684\u96c6\u7fa4\u3002\u53e6\u5916, \u5927\u90e8\u5206\u7684\u5c0f\u516c\u53f8\u5728\u8bad\u7ec3\u673a\u5668\u5b66\u4e60/\u6df1\u5ea6\u5b66\u4e60\u6a21\u578b\u7684\u65f6\u5019, \u5176\u5b9e\u662f\u4e0d\u9700\u8981\u5206\u5e03\u5f0f\u8bad\u7ec3\u7684\u3002\u4e00\u65b9\u9762\u56e0\u4e3a\u5c0f\u516c\u53f8\u7684\u6570\u636e\u91cf\u4e0d\u8db3\u4ee5\u8bad\u7ec3\u5206\u5e03\u5f0f\u7684\u5927\u6a21\u578b, \u53e6\u4e00\u65b9\u9762\u5206\u5e03\u5f0f\u6a21\u578b\u8bad\u7ec3\u662f\u4e00\u4e2a\u6bd4\u8f83\u590d\u6742\u7684\u5de5\u7a0b, \u5bf9\u5de5\u7a0b\u5e08\u7684\u8981\u6c42\u8f83\u9ad8, \u800c\u4e14\u670d\u52a1\u5668\u7684\u6210\u672c\u4e5f\u662f\u504f\u9ad8\u3002\u4f46\u662f, \u5982\u679c\u91c7\u7528\u5355\u673a\u8bad\u7ec3\u7684\u8bdd, \u5f80\u5f80\u4f1a\u51fa\u73b0Out-Of-Memory(OOM)\u548cOut-Of-Vocabulary(OOV)\u7684\u95ee\u9898\u3002Damo-Embedding\u5c31\u662f\u7528\u6765\u89e3\u51b3\u8fd9\u4e9b\u95ee\u9898\u7684\u9879\u76ee\u3002","sidebar":"tutorialSidebar"},"modules/CBF":{"id":"modules/CBF","title":"Counting Bloom Filter","description":"Why Counting Bloom Filter","sidebar":"tutorialSidebar"},"modules/Embedding":{"id":"modules/Embedding","title":"Embedding","description":"Embedding\u6a21\u5757\u4f7f\u7528rocksdb\u6765\u78c1\u76d8\u6765\u5b58\u50a8Embedding\u7684\u503c, \u91c7\u7528KV\u7684\u65b9\u5f0f\u3002 \u5176\u4e2dKey\u662f\u7279\u5f81hash\u7684\u503c(uint64\u7c7b\u578b), Value\u662fEmbedding\u5bf9\u5e94\u7684\u6d6e\u70b9\u6570\u5217\u8868\u4ee5\u53ca\u4e00\u4e9b\u5176\u4ed6\u7684\u503c\u3002","sidebar":"tutorialSidebar"},"modules/Initializer":{"id":"modules/Initializer","title":"Initializer","description":"When using an initializer, you need to configure the name item to indicate which initializer to use, and then configure their respective parameters according to different initializer. The following are the names of each initializer.","sidebar":"tutorialSidebar"},"modules/Modules":{"id":"modules/Modules","title":"Modules","description":"Modules","sidebar":"tutorialSidebar"},"modules/Optimizer":{"id":"modules/Optimizer","title":"Optimizer","description":"When using an optimizer, you need to configure the name item to indicate which optimizer to use, and then configure their respective parameters according to different optimizer. The following are the names of each optimizer.","sidebar":"tutorialSidebar"},"modules/Scheduler":{"id":"modules/Scheduler","title":"Learning Rate Scheduler","description":"When training deep neural networks, it is often useful to reduce learning rate as the training progresses. Learning rate schedules seek to adjust the learning rate during training by reducing the learning rate according to a pre-defined schedule.","sidebar":"tutorialSidebar"},"modules/Storage":{"id":"modules/Storage","title":"Storage","description":"We use rocksdb to save the data. You should create the storage object fisrt, then you can create embedding objects.","sidebar":"tutorialSidebar"},"quick_install":{"id":"quick_install","title":"\u5feb\u901f\u5b89\u88c5","description":"\u9700\u8981C++17\uff0c\u9700\u8981\u5347\u7ea7GCC/Clang (GCC >= 7, Clang >= 5).","sidebar":"tutorialSidebar"}}}')}}]);