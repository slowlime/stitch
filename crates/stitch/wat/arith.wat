(module $stitch-arith
        (func $stitch-specialize-i32
              (import "stitch" "specialize")
              (param $table-idx i32)
              (param $func-idx i32)
              (param $name-ptr i32)
              (param $name-len i32)
              (param $arg0 i32)
              (result i32))
        (func $stitch-specialize-i32-i32
              (import "stitch" "specialize")
              (param $table-idx i32)
              (param $func-idx i32)
              (param $name-ptr i32)
              (param $name-len i32)
              (param $arg0 i32)
              (param $arg1 i32)
              (result i32))
        (global $stitch-unknown-i32
                (import "stitch" "unknown")
                i32)

        (table $func-table (export "func-table") funcref
               (elem $add $fact))

        (memory $mem
                (data "add-2add-2-4fact-10"))

        (func $stitch-start (export "stitch-start")
              (drop
                (call
                  $stitch-specialize-i32-i32
                  (i32.const 0) ;; $func-table
                  (i32.const 0) ;; $add
                  (i32.const 0) ;; "add-2"
                  (i32.const 5)

                  (i32.const 2)
                  (global.get $stitch-unknown-i32)))

              (drop
                (call
                  $stitch-specialize-i32-i32
                  (i32.const 0) ;; $func-table
                  (i32.const 0) ;; $add
                  (i32.const 5) ;; "add-2-4"
                  (i32.const 7)

                  (i32.const 2)
                  (i32.const 4)))

              (drop
                (call
                  $stitch-specialize-i32
                  (i32.const 0) ;; $func-table
                  (i32.const 1) ;; $fact
                  (i32.const 12) ;; "fact-10"
                  (i32.const 7)

                  (i32.const 10))))

        (func $add (export "add") (param $lhs i32) (param $rhs i32) (result i32)
              (return
                (i32.add (local.get $lhs) (local.get $rhs))))

        (func $fact (export "fact") (param $n i32) (result i32)
              (return
                (if (result i32)
                  (i32.le_s (local.get $n) (i32.const 0))
                  (then (i32.const 1))
                  (else (i32.mul (local.get $n)
                                 (call $fact (i32.sub (local.get $n) (i32.const 1)))))))))
