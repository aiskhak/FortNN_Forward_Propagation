!------------------------------------------------------------------------
! FortFP ----------------------------------------------------------------
!    Fortran module that reads weights and biases -----------------------
!    and performs a forward propagation ---------------------------------
!------------------------------------------------------------------------

MODULE FortFP
   
   IMPLICIT NONE
   
   INTEGER, PARAMETER :: dp=selected_real_KIND(13)

   ! activation function type
   TYPE :: af
      PROCEDURE(ndarray), POINTER, NOPASS :: f
   END TYPE af

   ! array of function pointers for activation functions
   TYPE (af), DIMENSION(:), ALLOCATABLE :: activfuncs
   
   ! layer type
   TYPE :: layer
      
      REAL(kind=dp), DIMENSION (:,:), POINTER :: w, wd               ! matrix of weights
      REAL(kind=dp), DIMENSION (:), POINTER :: b, bd                 ! array of biases
      REAL(kind=dp), DIMENSION (:), POINTER :: y                     ! array of outputs
      PROCEDURE(ndarray), PUBLIC, NOPASS, POINTER :: af => NULL()    ! activation function
   
   END TYPE layer

   ! layer is declared as target (can be pointed efficiently)
   TARGET :: layer

   ! making pointers for working with input and output vectors
   REAL(kind=dp), DIMENSION (:), POINTER :: xp => NULL(), yp => NULL()
   
   ! neural network type: array of layers
   TYPE :: nn
      
      TYPE(layer), DIMENSION (:), ALLOCATABLE :: layers
   
   CONTAINS 
      
      ! initialize neural network
      PROCEDURE, PUBLIC :: init
      
      ! perform forward propagation
      PROCEDURE, PUBLIC :: query

   END TYPE nn
   
   ! dynamic arrays type
   TYPE :: dyn_arr

      REAL(kind=dp), DIMENSION(:), ALLOCATABLE :: v
      REAL(kind=dp), DIMENSION(:,:), ALLOCATABLE :: m

   END TYPE dyn_arr
   
   ! interface for activation functions
   ABSTRACT INTERFACE

      FUNCTION ndarray(n)
         
         IMPORT dp
         IMPORT af
         REAL(kind=dp), DIMENSION (:), INTENT(in) :: n
         REAL(kind=dp), DIMENSION (SIZE(n)) :: ndarray

      END FUNCTION ndarray

   END INTERFACE

   PRIVATE :: readwb, init, query

CONTAINS

   !=====================================================================
   ! initialization of neural network
   SUBROUTINE init(this, layers, activ_func)
      
      CLASS (nn), INTENT(inout) :: this

      ! array containing structure of layers, e.g., [100, 20, 30, 10]
      ! 100 input neurons; 20 neurons in first hidden layer;
      ! 30 neurons in second hidden layer; 10 neurons in output layer
      INTEGER, DIMENSION (:), INTENT(in) :: layers
      INTEGER :: i, nin, nout, nl

      ! list of activation functions passed in
      TYPE (af), DIMENSION(0:), INTENT(in) :: activ_func
      REAL(kind=dp) :: sd
      TYPE (dyn_arr) :: wb

      ! check if number of activation functions match number of layers
      IF (SIZE(activ_func) /= SIZE(layers)) THEN
         PRINT *, "The number of layers and activation functions do not match"
         STOP
      ENDIF
      
      ! number of layers and neurons
      nl = SIZE(layers)
      nin = layers(1)
      nout = layers(nl)
      
      ! layers' numbers begin from 0
      ALLOCATE (this%layers(0:nl-1))

      ! loop over layers
      DO i=1,nl-1
         
         ! allocate weights
         ALLOCATE (this%layers(i)%w(layers(i+1),layers(i)))
         ALLOCATE (this%layers(i)%wd(layers(i+1),layers(i)))

         ! allocate biases
         ALLOCATE (this%layers(i)%b(layers(i+1)))
         ALLOCATE (this%layers(i)%bd(layers(i+1)))

         ! set to 0
         this%layers(i)%wd = 0._dp
         this%layers(i)%bd = 0._dp

         ! allocate outputs
         ALLOCATE (this%layers(i)%y(layers(i+1)))

         ! initialize weights and biases using normal distribution
         ! based on Glorot method which is to force SD=sqrt(2/nin+nout) for each layer
         ! the standard deviation for creating the initial weights:
         sd = SQRT(2._dp / REAL(layers(i+1) + layers(i), dp))
         CALL readwb(i, SHAPE(this%layers(i)%w), wb)
         CALL readwb(i, SHAPE(this%layers(i)%b), wb)
         this%layers(i)%w = wb%m
         this%layers(i)%b = wb%v

         DEALLOCATE (wb%v,wb%m)

         ! assign activation functions
         this%layers(i)%af => activ_func(i)%f

      ENDDO

      ! manually create layer 0
      ALLOCATE (this%layers(0)%y(nin))

      ! no weights in layer 0
      ALLOCATE (this%layers(0)%w(0,0), this%layers(0)%wd(0,0)) 
      ALLOCATE (this%layers(0)%b(0), this%layers(0)%bd(0))

      ! activation function for layer 0 (very likely NULL)
      this%layers(0)%af => activ_func(0)%f

   END SUBROUTINE init
   
   !=====================================================================
   ! given input x, performs forward propagation with results in this%y
   SUBROUTINE query(this, x)

      CLASS(nn), INTENT(inout) :: this
      REAL(kind=dp), DIMENSION(:), INTENT(in) :: x
      INTEGER :: i, n, lm, ln
      INTEGER :: j, k
      REAL(kind=dp) :: sum_j
      
      ! number of weight matrices
      ! layer 0 does not have it, so the number is 1 unit less
      n = SIZE(this%layers) - 1
      
      ! copy x into y(0)
      this%layers(0)%y = x

      ! if activation function for inputs is used (very likely NULL)
      IF (ASSOCIATED(this%layers(0)%af)) &
           this%layers(0)%y = this%layers(0)%af (this%layers(0)%y)

      ! loop over layers
      DO i=1,n

         xp => this%layers(i-1)%y
         yp => this%layers(i)%y

         lm = UBOUND(this%layers(i)%w, 1)
         ln = UBOUND(this%layers(i)%w, 2)
         yp = this%layers(i)%b

         ! matrix-vector multiplication: lapack library
         !CALL dgemv("N", lm, ln, 1._dp, this%layers(i)%w, lm, xp, 1, 1._dp, yp, 1)  

         ! uncomment this staff if there is no access to lapack
         do k=1,lm
            sum_j = 0._dp
            do j=1,ln
               sum_j = sum_j + this%layers(i)%w(k,j)*xp(j)
            enddo
            yp(k) =  sum_j + this%layers(i)%b(k)
         enddo
         
         ! apply activation
         yp = this%layers(i)%af(yp)
         
      ENDDO


   END SUBROUTINE query

   ! =====================================================================
   ! read weights and biases from file
   SUBROUTINE readwb(n_skip, frm, arr)

      INTEGER, INTENT(in) :: n_skip
      TYPE (dyn_arr), INTENT(inout) :: arr
      REAL(kind=dp), DIMENSION(:), ALLOCATABLE :: nd
      INTEGER, DIMENSION(:), INTENT(in) :: frm
      INTEGER, PARAMETER :: ki = selected_int_KIND(18)
      INTEGER (kind=ki) :: samples, n, nm, osize
      INTEGER :: i, lm, ln, j, k, nsum

      ! nm = 2 for weights; nm = 1 for biases
      nm = SIZE(frm)
      
      ! allocate arrays for weights and open file
      IF (nm == 2) THEN
         samples = INT(frm(1), ki)*INT(frm(2), ki)
         osize = samples	! recording original size
         ! in case both dimensions are odd
         IF (MOD(samples, 2).EQ.1) samples = samples + 1
         ALLOCATE (arr%m(frm(1), frm(2)))
         OPEN (1111, FILE = 'wb/w_0', status = 'old')
      ! allocate arrays for biases
      ELSE
         samples=INT(frm(1),ki)
         osize = samples   	! recording original size
         ! in case dimension is odd
         IF (MOD(samples,2).EQ.1) samples = samples + 1
         ALLOCATE (arr%v(frm(1)))
         OPEN (1111, FILE = 'wb/b_0', status = 'old')
      ENDIF
      
      ! array for values
      ALLOCATE (nd(samples))
      nd = 0

      ! skip necessary lines
      IF (n_skip > 1) THEN
         REWIND (1111)
         DO i=1,n_skip-1
            READ (1111,*)
         END DO
      END IF
      ! read values from file
      READ (1111,*) nd
      CLOSE (1111)

      ! weights	
      IF (nm == 2) THEN
         ln = size(arr%m, dim=2)
         lm = size(arr%m, dim=1)
         nsum = 1
         do j=1,lm
            do k=1,ln
               arr%m(j,k) = nd(nsum)
               nsum = nsum + 1
            enddo
         enddo
      ! biases
      ELSE
         arr%v = nd(1:osize)
      ENDIF

   END SUBROUTINE readwb
   !=====================================================================

   !---------------------------------------------------------------------
   ! ACTIVATION FUNCTIONS -----------------------------------------------
   !    all inputs and outputs are multidimensional ---------------------
   !    so even a scalar must be with dimension (1) ---------------------
   !---------------------------------------------------------------------
   
   !=====================================================================  
   PURE FUNCTION sigmoid(x)
      
      ! sigmoid
      REAL(kind=dp), DIMENSION (:), INTENT(in) :: x
      REAL(kind=dp), DIMENSION (SIZE(x)) :: sigmoid
      
      sigmoid = EXP(-x)
      sigmoid = 1._dp + sigmoid
      sigmoid = 1._dp / sigmoid
   
   END FUNCTION sigmoid

   ! =====================================================================
   PURE FUNCTION htan (x)
      
      ! hyperbolic tan
      REAL(kind=dp), DIMENSION (:), INTENT(in) :: x
      REAL(kind=dp), DIMENSION (SIZE(x)) :: htan
      htan = TANH(x)

   END FUNCTION htan

   ! =====================================================================
   PURE FUNCTION smax (y)
      
      ! softmax
      REAL(kind=dp), DIMENSION(:), INTENT(in) :: y
      REAL(kind=dp), DIMENSION (SIZE(y)) :: smax
      REAL(kind=dp) :: sum_smax, l1, l2
      l1 = 1.e-9_dp
      l2 = 0.999999999_dp
      smax = EXP(y)
      sum_smax = SUM(smax)
      smax = smax / sum_smax
      WHERE (smax > l2) smax = l2
      WHERE (smax < l1) smax = l1

   END FUNCTION smax

   ! =====================================================================
   PURE FUNCTION lrelu(x)

      ! leaky relu
      REAL(kind=dp), DIMENSION (:), INTENT(in) :: x
      REAL(kind=dp), DIMENSION (SIZE(x)) :: lrelu
      lrelu = x
      WHERE (x < 0._dp) lrelu = 0.01_dp*x

   END FUNCTION lrelu

   ! =====================================================================
   PURE FUNCTION relu(x)
      
      ! relu
      REAL(kind=dp), DIMENSION (:), INTENT(in) :: x
      REAL(kind=dp), DIMENSION (SIZE(x)) :: relu

      relu = x
      WHERE ( x < 0._dp ) relu = 0._dp

   END FUNCTION relu
   ! =====================================================================

   PURE FUNCTION no_activation(x)
      
      ! no activation
      REAL(kind=dp), DIMENSION (:), INTENT(in) :: x
      REAL(kind=dp), DIMENSION (SIZE(x)) :: no_activation

      no_activation = x

   END FUNCTION no_activation
   ! =====================================================================

END MODULE FortFP
