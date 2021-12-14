# TypeScript

Typed JavaScript at Any Scale

TypeScript 是一门静态类型、弱类型的语言

## TypeScript 简要知识点

### 原始数据类型

-   布尔值：boolean

注意：使用构造函数 Boolean 创造的对象不是布尔值，是 boolean 对象。

-   数值类型：number

-   字符串类型：string

-   空值：void
    JavaScript 中没有空值的概念

-   null && undefined
    undefined 和 null 是所有类型的子类型，可以赋值给其他基础类型的变量

-   任意值：any
    用来表示允许赋值为任意类型
    在任意值上，允许**访问任何属性**，允许**调用任何方法**
    变量未声明类型，会被识别为 any 类型

-   未知值：unknown
    unknown 是 any 的安全版本，想用 any 时应该先试着用 unknown

### 类型推断

TypeScript 会在变量没有明确的指定类型时，给变量推测出一个类型
如果定义的时候没有赋值，不管之后有没有赋值，都会被推断成 any 类型而完全不被类型检查

### 对象类型

接口 `[interface]`

readonly
?
[]

type 创建`<类型别名>或者<字符串字面量类型>`

思考:interface 和 type 的区别

### 数组类型

有多种声明方式

### 函数类型

有多种声明方式

### 函数重载

### 类型断言

通过 as 手动指定一个值得类型

### 枚举：enum

### 联合类型：用 | 表示当前变量是多个类型的集合

### 类：class

TS 实现 ES 中类的用法

抽象类和抽象方法

类与接口
interface

### 泛型
