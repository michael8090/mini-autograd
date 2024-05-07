// use std::{borrow::BorrowMut, cell::RefCell, ops::{Add, AddAssign, Mul}, rc::Rc};

use std::{cell::RefCell, ops::{Add, Div, Mul, Sub}, rc::Rc};

enum Ops {
    Add,
    Mul,
    Pow,
    Relu,
}

#[derive(Clone)]
struct ValueHandle {
    idx: usize,
    builder: ValueBuilder,
}

struct Value {
    value: f32,
    handle: ValueHandle,
    grad: f32,
    ops: Option<Ops>,
    args: Option<Vec<ValueHandle>>
}

struct ValueStore {
    values: Vec<Value>,
}

impl ValueStore {
    fn new() -> Self {
        Self {values: vec![]}
    }

    fn get_value(&self, handle: &ValueHandle) -> &mut Value {
        unsafe {
            &mut *(&self.values[handle.idx] as *const _ as *mut Value)
        }
    }


    fn bfs(&self, entry_handle: ValueHandle, f: impl Fn(&ValueHandle)) {
        let mut queue: Vec<ValueHandle> = vec![entry_handle];

        while let Some(value_handle) = queue.pop() {
            let value = &self.values[value_handle.idx];
            f(&value.handle);

            value.args.clone().map(|args| queue.extend(args));
        }
    }

    fn zero_grads(&mut self) {
        for v in &mut self.values {
            v.grad = 0.0;
        }
    }

    fn backward(&mut self, entry_handle: ValueHandle) {
        let entry_value = self.get_value(&entry_handle);
        entry_value.grad = 1.0;
        self.bfs(entry_handle, |h| {
            let out_value = &self.get_value(h);
            match &out_value.ops {
                Some(ops) => {
                    match ops {
                        Ops::Add => {
                            let args = out_value.args.as_ref().unwrap();
                            let a = self.get_value(&args[0]);
                            let b = self.get_value(&args[1]);
                            a.grad += out_value.grad;
                            b.grad += out_value.grad;
                        },
                        Ops::Mul => {
                            let args = out_value.args.as_ref().unwrap();
                            let a = self.get_value(&args[0]);
                            let b = self.get_value(&args[1]);
                            a.grad += b.value * out_value.grad;
                            b.grad += a.value * out_value.grad;
                        },
                        Ops::Pow => {
                            let args = out_value.args.as_ref().unwrap();
                            let a = self.get_value(&args[0]);
                            let b = self.get_value(&args[1]);
                            a.grad += b.value * a.value.powf(b.value - 1.0) * out_value.grad;
                            b.grad += out_value.value * a.value.ln() * out_value.grad;
                        },
                        Ops::Relu => {
                            let args = out_value.args.as_ref().unwrap();
                            let a = self.get_value(&args[0]);
                            let grad = if a.value > 0.0 {1.0} else {0.0};
                            a.grad += grad * out_value.grad;
                        }
                    }
                }
                None => {
                    // panic!("a manually created value doesn't have derivative against any other variable");
                }
            }
        });
    }
}

impl Add<&ValueHandle> for &ValueHandle {
    type Output = ValueHandle;

    fn add(self, rhs:&ValueHandle) -> Self::Output {
        let out_data;
        {
            let store = self.builder.0.borrow();
            out_data = store.values[self.idx].value + store.values[rhs.idx].value;
        }
        let out = self.builder.value(out_data);
        {
            let store = self.builder.0.borrow();
            let out_value = store.get_value(&out);
            out_value.ops = Some(Ops::Add);
            out_value.args = Some(vec![self.clone(), rhs.clone()]);
        }

        out
    }
}

impl Mul<&ValueHandle> for &ValueHandle {
    type Output = ValueHandle;

    fn mul(self, rhs:&ValueHandle) -> Self::Output {
        let out_data;
        {
            let store = self.builder.0.borrow();
            out_data = store.values[self.idx].value * store.values[rhs.idx].value;
        }
        let out = self.builder.value(out_data);
        {
            let store = self.builder.0.borrow();
            let out_value = store.get_value(&out);
            out_value.ops = Some(Ops::Mul);
            out_value.args = Some(vec![self.clone(), rhs.clone()]);
        }

        out
    }
}

impl ValueHandle {
    fn pow(&self, rhs:&ValueHandle) -> Self {
        let out_data;
        {
            let store = self.builder.0.borrow();
            out_data = store.get_value(self).value.powf(store.get_value(rhs).value);
        }
        let out = self.builder.value(out_data);
        {
            let store = self.builder.0.borrow();
            let out_value = store.get_value(&out);
            out_value.ops = Some(Ops::Pow);
            out_value.args = Some(vec![self.clone(), rhs.clone()]);
        }

        out
    }

    fn relu(&self) -> Self {
        let out_data;
        {
            let store = self.builder.0.borrow();
            let self_data = store.get_value(self).value;
            out_data = self_data.max(0.0);
        }
        let out = self.builder.value(out_data);
        {
            let store = self.builder.0.borrow();
            let out_value = store.get_value(&out);
            out_value.ops = Some(Ops::Relu);
            out_value.args = Some(vec![self.clone()]);
        }

        out
    }
}

impl Sub<&ValueHandle> for &ValueHandle {
    type Output = ValueHandle;

    fn sub(self, rhs:&ValueHandle) -> Self::Output {
        self + &(&self.builder.value(-1.0) * rhs)
    }
}

impl Div<&ValueHandle> for &ValueHandle {
    type Output = ValueHandle;

    fn div(self, rhs:&ValueHandle) -> Self::Output {
        self * &(rhs.pow(&self.builder.value(-1.0)))
    }
}

#[derive(Clone)]
struct ValueBuilder (Rc<RefCell<ValueStore>>);

impl ValueBuilder {
    fn new() -> Self {
        let store = ValueStore::new();
        Self (Rc::new(RefCell::new(store)))
    }

    fn value(&self, v: f32) -> ValueHandle {
        let mut store = self.0.borrow_mut();
        let handle = ValueHandle{ idx: store.values.len(), builder: self.clone() };
        let ret = handle.clone();
        let value = Value{value: v, handle, grad: 0.0, ops: None, args: None};

        store.values.push(value);

        ret
    }
}

fn main() {
    let builder = ValueBuilder::new();
    let v1 = &builder.value(1.0);
    let v2 = &builder.value(2.0);
    let v3 = &builder.value(3.0);
    let a = (&(v1 / v2) * &(v3 - v2)).pow(&builder.value(10.0)).relu();
    let mut store = builder.0.borrow_mut();
    store.zero_grads();
    store.backward(a);
    println!("{:?}", store.get_value(v3).grad);
}
