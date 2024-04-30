// use std::{borrow::BorrowMut, cell::RefCell, ops::{Add, AddAssign, Mul}, rc::Rc};

use std::{cell::RefCell, ops::Add, rc::Rc};

enum Ops {
    Add,
    Mul,
    Pow,
    Relu,
}

type ValueIndex = usize;

struct Record {
    grad: f32,
    ops: Ops,
    args: Vec<ValueIndex>
}

struct Value {
    value: f32,
    idx: ValueIndex,
    builder: ValueBuilder,
    record: Option<Record>
}

struct ValueStore {
    values: Vec<Value>,
}

impl ValueStore {
    fn new() -> Self {
        Self {values: vec![]}
    }

    fn bfs(&mut self, entry_index: ValueIndex,f: impl Fn(&mut Value)) {
        let mut queue: Vec<ValueIndex> = vec![entry_index];

        while let Some(valueIndex) = queue.pop() {
            let value = &mut self.values[valueIndex];
            f(value);

            value.record.map(|r| queue.extend(r.args));
        }
    }

    fn zero_grads(&mut self) {
        for v in &mut self.values {
            if let Some(r) = &mut v.record {
                r.grad = 0.0;
            }
        }
    }

    fn backward(&mut self, entry_index: ValueIndex) {
        let entry_value = &mut self.values[entry_index];
        let mut record = entry_value.record.unwrap();
        record.grad = 1.0;
        self.bfs(entry_index, |value| {
            match value.record {
                Some(out_record) => {
                    match out_record.ops {
                        Ops::Add => {
                            let a = &mut self.values[out_record.args[0]];
                            let b = &mut self.values[out_record.args[1]];
                            a.record.unwrap().grad += out_record.grad;
                            b.record.unwrap().grad += out_record.grad;
                        },
                        Ops::Mul => {
                            let a = &mut self.values[out_record.args[0]];
                            let b = &mut self.values[out_record.args[1]];
                            a.record.unwrap().grad += b.value * out_record.grad;
                            b.record.unwrap().grad += a.value * out_record.grad;
                        },
                        Ops::Pow => {

                        },
                        Ops::Relu => {

                        }
                    }
                }
                None => {
                    panic!("a manually created value doesn't have derivative against any other variable");
                }
            }
        });
    }
}

impl<'a> Add<&'a Value> for &Value {
    type Output = &'a Value;

    fn add(self, rhs:&Value) -> Self::Output {
        let mut out = self.builder.value(self.value + rhs.value);
        out.record = Some(Record {
            grad: 0.0,
            ops: Ops::Add,
            args: vec![self.idx, rhs.idx],
        });

        out
    }
}

struct ValueBuilder (Rc<RefCell<ValueStore>>);

impl ValueBuilder {
    fn new() -> Self {
        let store = ValueStore::new();
        Self (Rc::new(RefCell::new(store)))
    }

    fn clone(&self) -> Self {
        Self(self.0.clone())
    }

    fn value(&mut self, v: f32) -> &Value {
        let mut store = self.0.borrow_mut();
        let idx = store.values.len();
        let value = Value{value: v, idx, builder: self.clone(), record: None};

        {let values = &mut store.values;
        values.push(value);}
        let values = &store.values;

        &values[idx]
    }
}

fn main() {
    let mut builder = ValueBuilder::new();
    let mut v1 = builder.value(1.0);
    let mut v2 = builder.value(2.0);
    let a = v1 + v2;
    println!("{:?}", a.value);
}
