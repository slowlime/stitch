(module $stitch-calc
  (func $stitch-arg-count
    (import "stitch" "arg-count")
    (result i32))
  (func $stitch-arg-len
    (import "stitch" "arg-len")
    (param $idx i32)
    (result i32))
  (func $stitch-arg-read
    (import "stitch" "arg-read")
    (param $idx i32)
    (param $buf i32)
    (param $size i32)
    (param $offset i32)
    (result i32))
  (func $stitch-specialize-i32
    (import "stitch" "specialize")
    (param $func-idx i32)
    (param $name-ptr i32)
    (param $name-len i32)
    (param $arg0 i32)
    (result i32))
  (func $stitch-const-ptr
    (import "stitch" "const-ptr")
    (param $ptr i32)
    (result i32))
  (func $stitch-propagate-load
    (import "stitch" "propagate-load")
    (param $ptr i32)
    (result i32))
  (func $stitch-print-value
    (import "stitch" "print-value")
    (param $value i64))
  (func $stitch-print-str
    (import "stitch" "print-str")
    (param $str i32)
    (param $len i32))
  (func $stitch-is-specializing
    (import "stitch" "is-specializing")
    (result i32))

  (table $func-table 16 funcref)
  (elem $func-table (i32.const 1) $interpret)

  (global $heap-end (mut i32) (i32.const 0x8000))
  (global $stack-ptr (mut i32) (i32.const 0x1000))

  (memory $mem 1)
  (data $mem (i32.const 0x1000) "specialized")
  (data $mem (i32.const 0x100b) "wrong number of arguments: expected 1")
  (data $mem (i32.const 0x1030) "not enough memory")
  (data $mem (i32.const 0x1041) "new capacity is less than current length")
  (data $mem (i32.const 0x1069) "trying to pop more bytes than available")
  (data $mem (i32.const 0x1090) "unexpected eof")
  (data $mem (i32.const 0x109e) "expected ')'")
  (data $mem (i32.const 0x10aa) "unexpected character")
  (data $mem (i32.const 0x10be) "illegal operation")
  (data $mem (i32.const 0x10cf) "value stack is empty or contains multiple integers")
  (data $mem (i32.const 0x1101) "i+-*/")

  (func $stitch-start (export "stitch-start")
    (local $sp i32)
    (local $prog-size i32)
    (local $prog-ptr i32)
    (local $specialized-idx i32)

    ;; stack layout:
    ;; @0 ops-vec: 12 bytes
    ;;   @0 buf: i32
    ;;   @4 len: i32
    ;;   @8 cap: i32

    (global.set $stack-ptr
      (local.tee $sp
        (i32.sub
          (global.get $stack-ptr)
          (i32.const 12))))

    (if
      (i32.ne
        (call $stitch-arg-count)
        (i32.const 1))
      (then
        (call $stitch-print-str
          (i32.const 0x100b)
          (i32.const 37))
        (unreachable)))

    (drop
      (call $stitch-arg-read
        (i32.const 0)
        (local.tee $prog-ptr
          (call $malloc
            (local.tee $prog-size
              (call $stitch-arg-len
                (i32.const 0)))))
        (local.get $prog-size)
        (i32.const 0)))
    (call $memset
      (local.get $sp)
      (i32.const 12)
      (i32.const 0))
    (call $parse-expr
      (local.get $prog-ptr)
      (local.get $prog-size)
      (local.get $sp))
    (local.set $specialized-idx
      (call $stitch-specialize-i32
        (i32.const 1) ;; $interpret
        (i32.const 0) ;; "specialized"
        (i32.const 11)
        (call $stitch-propagate-load
          (call $stitch-const-ptr
            (local.get $sp)))))
    (call $stitch-print-value
      (call_indirect (result i64)
        (local.get $specialized-idx))))

  (func $print-str (param $str i32) (param $len i32)
    (if
      (call $stitch-is-specializing)
      (then
        (call $stitch-print-str
          (local.get $str)
          (local.get $len)))))

  (func $malloc (param $size i32) (result i32)
    (local $new-size i32)
    (local $ptr i32)
    (local $extra-pages i32)

    (if $grow-mem
      (local.tee $extra-pages
        (i32.sub
          (i32.shr_u
            (i32.add
              (local.tee $new-size
                (i32.add
                  (local.tee $ptr
                    (global.get $heap-end))
                  (local.get $size)))
              (i32.const 65535))
            (i32.const 16))
          (memory.size)))
      (then
        (br_if $grow-mem
          (i32.ne
            (memory.grow
              (local.get $extra-pages))
            (i32.const -1)))
        ;; not enough memory
        (call $print-str
          (i32.const 0x1030)
          (i32.const 17))
        (unreachable)))

    (global.set $heap-end
      (local.get $new-size))
    (return
      (local.get $ptr)))

  (func $memset (param $ptr i32) (param $size i32) (param $byte i32)
    (block $body
      (loop $loop
        (br_if $body
          (i32.eqz
            (local.get $size)))
        (i32.store8
          (local.get $ptr)
          (local.get $byte))
        (local.set $ptr
          (i32.add
            (local.get $ptr)
            (i32.const 1)))
        (local.set $size
          (i32.sub
            (local.get $size)
            (i32.const 1)))
        (br $loop))))

  (func $memcpy (param $dst i32) (param $src i32) (param $size i32)
    (block $body
      (loop $loop
        (br_if $body
          (i32.eqz
            (local.get $size)))
        (i32.store8
          (local.get $dst)
          (i32.load8_u
            (local.get $src)))
        (local.set $dst
          (i32.add
            (local.get $dst)
            (i32.const 1)))
        (local.set $src
          (i32.add
            (local.get $src)
            (i32.const 1)))
        (local.set $size
          (i32.sub
            (local.get $size)
            (i32.const 1)))
        (br $loop))))

  (func $vec-len (param $vec i32) (result i32)
    (i32.load offset=4
      (local.get $vec)))

  (func $vec-cap (param $vec i32) (result i32)
    (i32.load offset=8
      (local.get $vec)))

  (func $vec-resize (param $vec i32) (param $new-cap i32)
    (local $new-buf i32)

    (if
      (i32.lt_u
        (local.get $new-cap)
        (call $vec-len
          (local.get $vec)))
      (then
        (call $print-str
          (i32.const 0x1041)
          (i32.const 40))
        (unreachable)))

    (call $memcpy
      (local.tee $new-buf
        (call $malloc
          (local.get $new-cap)))
      (i32.load
        (local.get $vec))
      (call $vec-len
        (local.get $vec)))

    (i32.store
      (local.get $vec)
      (local.get $new-buf))
    (i32.store offset=8
      (local.get $vec)
      (local.get $new-cap)))

  (func $vec-push (param $vec i32) (param $size i32) (result i32)
    (local $new-len i32)
    (local $old-len i32)

    (if
      (i32.gt_u
        (local.tee $new-len
          (i32.add
            (local.tee $old-len
              (call $vec-len
                (local.get $vec)))
            (local.get $size)))
        (call $vec-cap
          (local.get $vec)))
      (then
        (call $vec-resize
          (local.get $vec)
          ;; 1 << log2($new-len)
          (i32.shl
            (i32.const 1)
            (i32.sub
              (i32.const 32)
              (i32.clz
                (i32.sub
                  (local.get $new-len)
                  (i32.const 1))))))))

    (i32.store offset=4
      (local.get $vec)
      (local.get $new-len))
    (i32.add
      (i32.load
        (local.get $vec))
      (local.get $old-len)))

  (func $vec-pop (param $vec i32) (param $size i32)
    (local $old-len i32)

    (if
      (i32.lt_u
        (local.tee $old-len
          (call $vec-len
            (local.get $vec)))
        (local.get $size))
      (then
        (call $print-str
          (i32.const 0x1069)
          (i32.const 39))
        (unreachable)))

    (i32.store offset=4
      (local.get $vec)
      (i32.sub
        (local.get $old-len)
        (local.get $size))))

  (func $vec-pop-i64 (param $vec i32) (result i64)
    (local $result i64)
    (local $old-len i32)
    (local $new-len i32)

    (if
      (i32.lt_u
        (local.tee $old-len
          (call $vec-len
            (local.get $vec)))
        (i32.const 8))
      (then
        (call $print-str
          (i32.const 0x1069)
          (i32.const 39))
        (unreachable)))

    (local.set $result
      (i64.load
        (i32.add
          (i32.load
            (local.get $vec))
          (local.tee $new-len
            (i32.sub
              (local.get $old-len)
              (i32.const 8))))))
    (i32.store offset=4
      (local.get $vec)
      (local.get $new-len))

    (local.get $result))

  (func $parse-expr (param $buf i32) (param $size i32) (param $ops-vec i32)
    (local $sp i32)

    ;; stack layout:
    ;; @0 parser: 12 bytes
    ;;   @0 buf: i32
    ;;   @4 size: i32
    ;;   @8 ops-vec: i32

    (global.set $stack-ptr
      (local.tee $sp
        (i32.sub
          (global.get $stack-ptr)
          (i32.const 12))))
    (i32.store
      (local.get $sp)
      (local.get $buf))
    (i32.store offset=4
      (local.get $sp)
      (local.get $size))
    (i32.store offset=8
      (local.get $sp)
      (local.get $ops-vec))

    (call $parse-root
      (local.get $sp))

    (global.set $stack-ptr
      (i32.add
        (local.get $sp)
        (i32.const 12))))

  (func $parser-is-eof (param $parser i32) (result i32)
    (i32.eqz
      (i32.load offset=4
        (local.get $parser))))

  (func $parser-peek (param $parser i32) (result i32)
    (i32.load8_u
      (i32.load
        (local.get $parser))))

  (func $parser-next (param $parser i32)
    (i32.store
      (local.get $parser)
      (i32.add
        (i32.load
          (local.get $parser))
        (i32.const 1)))
    (i32.store offset=4
      (local.get $parser)
      (i32.sub
        (i32.load offset=4
          (local.get $parser))
        (i32.const 1))))

  (func $parser-try-consume (param $parser i32) (param $expected i32) (result i32)
    (if
      (i32.eq
        (call $parser-peek
          (local.get $parser))
        (local.get $expected))
      (then
        (call $parser-next
          (local.get $parser))
        (return (i32.const 1))))

    (i32.const 0))

  (func $skip-whitespace (param $parser i32)
    (block $body
      (loop $loop
        (br_if $body
          (call $parser-is-eof
            (local.get $parser)))
        (br_if $loop
          (call $parser-try-consume
            (local.get $parser)
            ;; ' '
            (i32.const 32)))
        (br_if $loop
          (call $parser-try-consume
            (local.get $parser)
            ;; '\n'
            (i32.const 10)))
        (br_if $loop
          (call $parser-try-consume
            (local.get $parser)
            ;; '\r'
            (i32.const 13)))
        (br_if $loop
          (call $parser-try-consume
            (local.get $parser)
            ;; '\t'
            (i32.const 9))))))

  (func $parse-root (param $parser i32)
    (call $parse-add-sub
      (local.get $parser)))

  (func $parse-add-sub (param $parser i32)
    (local $op i32)

    (block $body
      (call $parse-mul-div
        (local.get $parser))

      (loop $loop
        (call $skip-whitespace
          (local.get $parser))
        (br_if $body
          (call $parser-is-eof
            (local.get $parser)))

        (if
          (call $parser-try-consume
            (local.get $parser)
            ;; '+'
            (i32.const 43))
          (then
            (local.set $op
              ;; OP_ADD
              (i32.const 1)))
          (else
            (if
              (call $parser-try-consume
                (local.get $parser)
                ;; '-'
                (i32.const 45))
              (then
                (local.set $op
                  ;; OP_SUB
                  (i32.const 2)))
              (else
                (return)))))

        (call $parse-mul-div
          (local.get $parser))
        (i32.store8
          (call $vec-push
            (i32.load offset=8
              (local.get $parser))
            (i32.const 1))
          (local.get $op))
        (br $loop))))

  (func $parse-mul-div (param $parser i32)
    (local $op i32)

    (block $body
      (call $parse-atom
        (local.get $parser))

      (loop $loop
        (call $skip-whitespace
          (local.get $parser))
        (br_if $body
          (call $parser-is-eof
            (local.get $parser)))

        (if
          (call $parser-try-consume
            (local.get $parser)
            ;; '*'
            (i32.const 42))
          (then
            (local.set $op
              ;; OP_MUL
              (i32.const 3)))
          (else
            (if
              (call $parser-try-consume
                (local.get $parser)
                ;; '/'
                (i32.const 47))
              (then
                (local.set $op
                  ;; OP_DIV
                  (i32.const 4)))
              (else
                (return)))))

        (call $parse-atom
          (local.get $parser))
        (i32.store8
          (call $vec-push
            (i32.load offset=8
              (local.get $parser))
            (i32.const 1))
          (local.get $op))
        (br $loop))))

  (func $parse-atom (param $parser i32)
    (local $chr i32)

    (block $body
      (call $skip-whitespace
        (local.get $parser))

      (if
        (call $parser-is-eof
          (local.get $parser))
        (then
          (call $print-str
            (i32.const 0x1090)
            (i32.const 14))
          (unreachable)))

      (if
        (call $parser-try-consume
          (local.get $parser)
          ;; '('
          (i32.const 40))
        (then
          (call $parse-root
            (local.get $parser))
          (br_if $body
            (call $parser-try-consume
              (local.get $parser)
              ;; ')'
              (i32.const 41)))
          (call $print-str
            (i32.const 0x109e)
            (i32.const 12))
          (unreachable)))

      (if
        (i32.eq
          (local.tee $chr
            (call $parser-peek
              (local.get $parser)))
          ;; '-'
          (i32.const 45))
        (then
          (call $parse-int
            (local.get $parser))
          (return)))

      (if
        (call $is-digit
          (local.get $chr))
        (then
          (call $parse-int
            (local.get $parser))
          (return)))

      (call $print-str
        (i32.const 0x10aa)
        (i32.const 20))
      (unreachable)))

  (func $is-digit (param $chr i32) (result i32)
    (i32.and
      (i32.ge_u
        (local.get $chr)
        ;; '0'
        (i32.const 48))
      (i32.le_u
        (local.get $chr)
        ;; '9'
        (i32.const 57))))

  (func $parse-int (param $parser i32)
    (local $neg i32)
    (local $result i64)
    (local $consumed i32)
    (local $chr i32)
    (local $ptr i32)

    (if
      (call $parser-try-consume
        (local.get $parser)
        ;; '-'
        (i32.const 45))
      (then
        (local.set $neg
          (i32.const 1))))

    (block $read-digits
      (loop $read-digits-loop
        (br_if $read-digits
          (call $parser-is-eof
            (local.get $parser)))
        (br_if $read-digits
          (i32.eqz
            (call $is-digit
              (local.tee $chr
                (call $parser-peek
                  (local.get $parser))))))
        (call $parser-next
          (local.get $parser))
        (local.set $consumed
          (i32.const 1))
        (local.set $result
          (i64.add
            (i64.mul
              (local.get $result)
              (i64.const 10))
            (i64.extend_i32_u
              (i32.sub
                (local.get $chr)
                ;; '0'
                (i32.const 48)))))
        (br $read-digits-loop)))

    (if
      (local.get $consumed)
      (then
        (if
          (local.get $neg)
          (then
            (local.set $result
              (i64.sub
                (i64.const 0)
                (local.get $result)))))

        (i32.store8
          (local.tee $ptr
            (call $vec-push
              (i32.load offset=8
                (local.get $parser))
              (i32.const 9)))
          ;; OP_INT
          (i32.const 0))
        (i64.store offset=1
          (local.get $ptr)
          (local.get $result)))
      (else
        (call $print-str
          (i32.const 0x10aa)
          (i32.const 20))
        (unreachable))))

  (func $interpret (param $ops-vec i32) (result i64)
    (local $sp i32)
    (local $pc i32)
    (local $op i32)
    (local $lhs i64)
    (local $rhs i64)

    ;; stack layout:
    ;; @0 stack: 12 bytes
    ;;   @0 buf: i32
    ;;   @4 len: i32
    ;;   @8 cap: i32

    (global.set $stack-ptr
      (local.tee $sp
        (i32.sub
          (global.get $stack-ptr)
          (i32.const 12))))
    (call $memset
      (local.get $sp)
      (i32.const 12)
      (i32.const 0))

    (block $process-ops
      (loop $loop
        (block $div
          (block $mul
            (block $sub
              (block $add
                (block $int
                  (block $dispatch
                    (br_if $process-ops
                      (i32.ge_u
                        (local.get $pc)
                        (call $vec-len
                          (local.get $ops-vec))))
                    (local.set $op
                      (i32.load8_u
                        (i32.add
                          (i32.load
                            (local.get $ops-vec))
                          (local.get $pc))))
                    (call $print-str
                      (i32.add
                        (i32.const 0x1101)
                        (local.get $op))
                      (i32.const 1))
                    (local.set $pc
                      (i32.add
                        (local.get $pc)
                        (i32.const 1)))
                    (br_table $dispatch $int $add $sub $mul $div
                      (local.get $op)))

                  ;; OP_INT
                  (call $stitch-print-value
                    (local.tee $lhs
                      (i64.load
                        (i32.add
                          (i32.load
                            (local.get $ops-vec))
                          (local.get $pc)))))
                  (i64.store
                    (call $vec-push
                      (local.get $sp)
                      (i32.const 8))
                    (local.get $lhs))
                  (local.set $pc
                    (i32.add
                      (local.get $pc)
                      (i32.const 8)))
                  (br $loop))

                ;; OP_ADD
                (local.set $rhs
                  (call $vec-pop-i64
                    (local.get $sp)))
                (local.set $lhs
                  (call $vec-pop-i64
                    (local.get $sp)))
                (i64.store
                  (call $vec-push
                    (local.get $sp)
                    (i32.const 8))
                  (i64.add
                    (local.get $lhs)
                    (local.get $rhs)))
                (br $loop))

              ;; OP_SUB
              (local.set $rhs
                (call $vec-pop-i64
                  (local.get $sp)))
              (local.set $lhs
                (call $vec-pop-i64
                  (local.get $sp)))
              (i64.store
                (call $vec-push
                  (local.get $sp)
                  (i32.const 8))
                (i64.sub
                  (local.get $lhs)
                  (local.get $rhs)))
              (br $loop))

            ;; OP_MUL
            (local.set $rhs
              (call $vec-pop-i64
                (local.get $sp)))
            (local.set $lhs
              (call $vec-pop-i64
                (local.get $sp)))
            (i64.store
              (call $vec-push
                (local.get $sp)
                (i32.const 8))
              (i64.mul
                (local.get $lhs)
                (local.get $rhs)))
            (br $loop))

          ;; OP_DIV
          (local.set $rhs
            (call $vec-pop-i64
              (local.get $sp)))
          (local.set $lhs
            (call $vec-pop-i64
              (local.get $sp)))
          (i64.store
            (call $vec-push
              (local.get $sp)
              (i32.const 8))
            (i64.div_s
              (local.get $lhs)
              (local.get $rhs)))
          (br $loop))

        ;; illegal operation
        (call $print-str
          (i32.const 0x10be)
          (i32.const 17))
        (unreachable)))

    (if
      (i32.ne
        (call $vec-len
          (local.get $sp))
        (i32.const 8))
      (then
        ;; the stack is empty or contains more than one integer
        (call $print-str
          (i32.const 0x10cf)
          (i32.const 50))
        (unreachable)))

    (global.set $stack-ptr
      (i32.add
        (local.get $sp)
        (i32.const 12)))
    (i64.load
      (i32.load
        (local.get $sp)))))
