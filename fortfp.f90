!------------------------------------------------------------------------
! fortfp ----------------------------------------------------------------
!    Fortran module that reads weights and biases -----------------------
!    and performs a forward propagation ---------------------------------
!------------------------------------------------------------------------

module fortfp
   
   implicit none
   
   ! activation function type
   type :: af
      procedure(ndarray), pointer, nopass :: f
   end type af

   ! array of function pointers for activation functions
   type (af), dimension(:), allocatable :: activfuncs
   
   ! layer type
   type :: layer
      
      real(8), dimension (:,:), pointer :: w, wd               		! matrix of weights
      real(8), dimension (:), pointer :: b, bd                 		! array of biases
      real(8), dimension (:), pointer :: y                     		! array of outputs
      procedure(ndarray), public, nopass, pointer :: af => null()	! activation function
   
   end type layer

   ! making pointers for working with input and output vectors
   real(8), dimension (:), pointer :: xp => null(), yp => null()
   
   ! neural network type: array of layers
   type :: nn
      
      type(layer), dimension (:), allocatable :: layers
   
   contains 
      
      ! initialize neural network
      procedure, public :: init
      
      ! perform forward propagation
      procedure, public :: query

   end type nn
   
   ! dynamic arrays type
   type :: dyn_arr

      real(8), dimension(:), allocatable :: v
      real(8), dimension(:,:), allocatable :: m

   end type dyn_arr
   
   ! interface for activation functions
   abstract interface

      function ndarray(n)
         
         import af
         real(8), dimension (:), intent(in) :: n
         real(8), dimension (size(n)) :: ndarray

      end function ndarray

   end interface

   private :: readwb, init, query

contains

   !=====================================================================
   ! initialization of neural network
   subroutine init(this, layers, activ_func)
      
      class (nn), intent(inout) :: this

      ! array containing structure of layers, e.g., [100, 20, 30, 10]
      ! 100 input neurons; 20 neurons in first hidden layer;
      ! 30 neurons in second hidden layer; 10 neurons in output layer
      integer, dimension (:), intent(in) :: layers
      integer :: i, nin, nout, nl

      ! list of activation functions passed in
      type (af), dimension(0:), intent(in) :: activ_func
      real(8) :: sd
      type (dyn_arr) :: wb

      ! check if number of activation functions match number of layers
      if (size(activ_func) /= size(layers)) then
         print *, "The number of layers and activation functions do not match"
         stop
      endif
      
      ! number of layers and neurons
      nl = size(layers)
      nin = layers(1)
      nout = layers(nl)
      
      ! layers' numbers begin from 0
      allocate (this%layers(0:nl-1))

      ! loop over layers
      do i=1,nl-1
         
         ! allocate weights
         allocate (this%layers(i)%w(layers(i+1),layers(i)))
         allocate (this%layers(i)%wd(layers(i+1),layers(i)))

         ! allocate biases
         allocate (this%layers(i)%b(layers(i+1)))
         allocate (this%layers(i)%bd(layers(i+1)))

         ! set to 0
         this%layers(i)%wd = 0.d0
         this%layers(i)%bd = 0.d0

         ! allocate outputs
         allocate (this%layers(i)%y(layers(i+1)))

         ! read weights and biases
         call readwb(i, shape(this%layers(i)%w), wb)
         call readwb(i, shape(this%layers(i)%b), wb)
         this%layers(i)%w = wb%m
         this%layers(i)%b = wb%v

         deallocate (wb%v,wb%m)

         ! assign activation functions
         this%layers(i)%af => activ_func(i)%f

      enddo

      ! manually create layer 0
      allocate (this%layers(0)%y(nin))

      ! no weights in layer 0
      allocate (this%layers(0)%w(0,0), this%layers(0)%wd(0,0)) 
      allocate (this%layers(0)%b(0), this%layers(0)%bd(0))

      ! activation function for layer 0 (very likely null)
      this%layers(0)%af => activ_func(0)%f

   end subroutine init
   
   !=====================================================================
   ! given input x, performs forward propagation with results in this%y
   subroutine query(this, x)

      class(nn), intent(inout) :: this
      real(8), dimension(:), intent(in) :: x
      integer :: i, n, lm, ln
      integer :: j, k
      real(8) :: sum_j
      
      ! number of weight matrices
      ! layer 0 does not have it, so the number is 1 unit less
      n = size(this%layers) - 1
      
      ! copy x into y(0)
      this%layers(0)%y = x

      ! if activation function for inputs is used (very likely null)
      if (associated(this%layers(0)%af)) &
           this%layers(0)%y = this%layers(0)%af (this%layers(0)%y)

      ! loop over layers
      do i=1,n

         xp => this%layers(i-1)%y
         yp => this%layers(i)%y

         lm = ubound(this%layers(i)%w, 1)
         ln = ubound(this%layers(i)%w, 2)
         yp = this%layers(i)%b

         ! matrix-vector multiplication: lapack library
         !call dgemv("N", lm, ln, 1.d0, this%layers(i)%w, lm, xp, 1, 1.d0, yp, 1)  

         ! uncomment this staff if there is no access to lapack
         do k=1,lm
            sum_j = 0.d0
            do j=1,ln
               sum_j = sum_j + this%layers(i)%w(k,j)*xp(j)
            enddo
            yp(k) =  sum_j + this%layers(i)%b(k)
         enddo
         
         ! apply activation
         yp = this%layers(i)%af(yp)
         
      enddo

   end subroutine query

   ! =====================================================================
   ! read weights and biases from file
   subroutine readwb(n_skip, frm, arr)

      integer, intent(in) :: n_skip
      type (dyn_arr), intent(inout) :: arr
      real(8), dimension(:), allocatable :: nd
      integer, dimension(:), intent(in) :: frm
      integer(4) :: samples, n, nm, osize
      integer(4) :: i, lm, ln, j, k, nsum

      ! nm = 2 for weights; nm = 1 for biases
      nm = size(frm)
      
      ! allocate arrays for weights and open file
      if (nm == 2) then
         samples = int(frm(1))*int(frm(2))
         osize = samples	! recording original size
         ! in case both dimensions are odd
         if (mod(samples, 2) == 1) samples = samples + 1
         allocate (arr%m(frm(1), frm(2)))
         open (1111, file = 'wb/w_0', status = 'old')
      ! allocate arrays for biases
      else
         samples=int(frm(1))
         osize = samples   	! recording original size
         ! in case dimension is odd
         if (mod(samples,2) == 1) samples = samples + 1
         allocate (arr%v(frm(1)))
         open (1111, file = 'wb/b_0', status = 'old')
      endif
      
      ! array for values
      allocate (nd(samples))
      nd = 0

      ! skip necessary lines
      if (n_skip > 1) then
         rewind (1111)
         do i=1,n_skip-1
            read (1111,*)
         end do
      end if
      ! read values from file
      read (1111,*) nd
      close (1111)

      ! weights
      if (nm == 2) then
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
      else
         arr%v = nd(1:osize)
      endif

   end subroutine readwb
   !=====================================================================

   !---------------------------------------------------------------------
   ! ACTIVATION FUNCTIONS -----------------------------------------------
   !    all inputs and outputs are multidimensional ---------------------
   !    so even a scalar must be with dimension (1) ---------------------
   !---------------------------------------------------------------------
   
   !=====================================================================  
   pure function sigmoid(x)
      
      ! sigmoid
      real(8), dimension (:), intent(in) :: x
      real(8), dimension (size(x)) :: sigmoid
      
      sigmoid = exp(-x)
      sigmoid = 1.d0 + sigmoid
      sigmoid = 1.d0 / sigmoid
   
   end function sigmoid

   ! =====================================================================
   pure function htan (x)
      
      ! hyperbolic tan
      real(8), dimension (:), intent(in) :: x
      real(8), dimension (size(x)) :: htan
      htan = tanh(x)

   end function htan

   ! =====================================================================
   pure function smax (y)
      
      ! softmax
      real(8), dimension(:), intent(in) :: y
      real(8), dimension (size(y)) :: smax
      real(8) :: sum_smax, l1, l2
      l1 = 1.d-9
      l2 = 0.999999999d0
      smax = exp(y)
      sum_smax = sum(smax)
      smax = smax / sum_smax
      where (smax > l2) smax = l2
      where (smax < l1) smax = l1

   end function smax

   ! =====================================================================
   pure function lrelu(x)

      ! leaky relu
      real(8), dimension (:), intent(in) :: x
      real(8), dimension (size(x)) :: lrelu
      lrelu = x
      where (x < 0.d0) lrelu = 0.01d0*x

   end function lrelu

   ! =====================================================================
   pure function relu(x)
      
      ! relu
      real(8), dimension (:), intent(in) :: x
      real(8), dimension (size(x)) :: relu

      relu = x
      where ( x < 0.d0 ) relu = 0.d0

   end function relu
   ! =====================================================================

   pure function no_activation(x)
      
      ! no activation
      real(8), dimension (:), intent(in) :: x
      real(8), dimension (size(x)) :: no_activation

      no_activation = x

   end function no_activation
   ! =====================================================================

end module fortfp
