# CLASS ONE：回顾 Web 基础

## HTML

-   超文本标记语言（HuperText Markup Language）

-   块级元素和行内元素

    -   块级元素：单独占据一行
    -   行内元素：和其他元素公用一行

-   语义化

1. 便于开发者阅读，SEO 优化
2. 在没有 CSS 的情况下也能很好展现内容结构
3. 增强用户体验，也便于其他设备解析（屏幕阅读器，移动设备等）

-   特殊标签
    -   `<!DOCTYPE>`不是一个 HTML 标签，是用来告诉浏览器使用了哪个 HTML 版本。现在已经过时了。
    -   `<meta>`标签是用来描述 HTML 文档的元数据，不会显示在网页上，但会被浏览器解析。

## CSS

&emsp;&emsp;CSS 是用来指定文档如何展示给用户的一门语言——如网页的样式、布局等等。<br/>
&emsp;&emsp;结构包括一个选择器、花括号里面包着 key-value 结构的样式规则等。<br/>

-   内联在标签属性里面
-   内敛在 style 标签里面
-   link 外部 CSS 文件

-   CSS 的选择器种类

    -   ID 选择器：如 #id {}
    -   类选择器：如 .class {}
    -   属性选择器：如 a [href="..."] {}
    -   伪类选择器：如 :hover {}
    -   标签选择器：如 span {}
    -   伪元素选择器：如 ::before {}
    -   通配符选择器：如 \* {}
    -   多个选择器组合：如 p.class#id:hover {}

-   盒模型

    -   fix 定位
    -   relative 定位
    -   absolute 定位

-   布局方式

    -   Flex 布局
        Flexible Box 模型
    -   栅格布局
        将网页分为 n 行 n 列，规定每一部分在第几行第几列

## JavaScript

&emsp;&emsp;脚本语言，可以在网页上实现复杂的功能。<br/>

-   执行上下文（execution context）：当 JavaScript 代码执行一段可执行代码（execution code）时，会创建对应的执行上下文（execution context）。
-   闭包
-   构造函数和原型

    -   原型链和继承
        不同的对象通过 **proto** 连接在一起，形成了一条链，这就是原型链。

-   DOM
    DOM 是浏览器提供封装给 JS 的一系列操作 DOM 元素的 API，是实现网站动态效果的核心。

# CLASS TWO：学习 ES6

安全性，易用性，无歧义

-   ES6 标准
-   块级作用域
-   let 和 const
-   模板字符串
-   解构赋值
-   拓展运算符和 rest
-   可选链和 null 判断符
-   箭头函数

    -   this 指向上一层作用域

-   Set 数据结构
-   Map 数据结构

-   异步
    与同步处理相对，异步处理不用阻塞当前线程来等待处理完成，而是允许后续操作，直至其它线程将处理完成，并回调通知此线程。
    -   多线程
    -   回调函数
    -   Promise 状态机：pending fullfilled rejected
        Promise.all 和 Promise.race
    -   Async / await
    -   class
    -   extends
    -   Object.defineProperty 修改对象的属性
        适用场景：MVVM
    -   Proxy——Vue3 和 Mobx5 中的核心 API
        可以对目标对象进行拦截，外界对这个对象的访问和修改都能被拦截到
        适用场景：表单校验、错误屏蔽
    -   module
        require, exports
        import, export
        export default
